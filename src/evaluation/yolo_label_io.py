from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


# YOLO 라벨을 이미지와 매칭할 때 흔히 만나는 확장자를 모두 허용한다.
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".WEBP",
)


@dataclass(frozen=True)
class ImageInfo:
    """라벨 해석에 필요한 이미지 메타데이터를 한곳에 묶어 둔 구조체다."""

    stem: str
    file_name: str
    path: Path
    width: int
    height: int


def resolve_yolo_split_dirs(split_path: str | Path) -> tuple[Path, Path]:
    """YOLO split 경로를 `(images_dir, labels_dir)` 형태로 정규화한다.

    프로젝트 안에서는 다음 세 가지 형태가 모두 들어올 수 있다.
    1. split 루트 자체: `.../test`
    2. 이미지 폴더 직접 지정: `.../test/images`
    3. 라벨 폴더 직접 지정: `.../test/labels`
    4. 이미지 split 폴더 직접 지정: `.../images/test`
    5. 라벨 split 폴더 직접 지정: `.../labels/test`

    호출하는 쪽이 어떤 형태를 넘기든, 여기서 실제 이미지/라벨 폴더 쌍으로
    바꿔 주면 이후 로직은 동일한 규칙만 믿고 동작할 수 있다.
    """
    path = Path(split_path)

    if path.name == "images":
        images_dir = path
        labels_dir = path.parent / "labels"
    elif path.name == "labels":
        images_dir = path.parent / "images"
        labels_dir = path
    elif path.parent.name == "images":
        images_dir = path
        labels_dir = path.parents[1] / "labels" / path.name
    elif path.parent.name == "labels":
        images_dir = path.parents[1] / "images" / path.name
        labels_dir = path
    else:
        images_dir = path / "images"
        labels_dir = path / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")
    return images_dir.resolve(), labels_dir.resolve()


def resolve_prediction_labels_dir(path: str | Path) -> Path:
    """예측 결과 경로를 실제 `labels` 폴더 경로로 맞춘다.

    예측 결과를 받을 때 어떤 코드는 `.../prediction`을, 어떤 코드는
    `.../prediction/labels`를 넘긴다. 두 경우를 모두 허용하되, 실제로는
    항상 txt 파일이 들어 있는 `labels` 폴더를 반환하도록 통일한다.
    """
    raw = Path(path)
    labels_dir = raw if raw.name == "labels" else raw / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"prediction labels dir not found: {labels_dir}")
    return labels_dir.resolve()


def build_image_index(images_dir: str | Path) -> dict[str, ImageInfo]:
    """이미지 stem 기준 인덱스를 만든다.

    YOLO txt 파일은 보통 `IMG_0001.txt`처럼 stem 기준으로 이미지와 연결된다.
    따라서 overlay/evaluation 단계에서는 `stem -> 이미지 정보` 매핑을 미리
    만들어 두면, 각 txt 파일을 읽을 때마다 이미지 폴더를 다시 훑지 않아도 된다.
    """
    images_dir = Path(images_dir)
    index: dict[str, ImageInfo] = {}

    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix not in IMAGE_EXTENSIONS:
            continue

        # YOLO 정규화 좌표를 실제 픽셀 좌표로 복원하려면 원본 크기가 필요하다.
        with Image.open(path) as image:
            width, height = image.size

        index[path.stem] = ImageInfo(
            stem=path.stem,
            file_name=path.name,
            path=path.resolve(),
            width=width,
            height=height,
        )

    if not index:
        raise ValueError(f"no images found in {images_dir}")
    return index


def load_yolo_label_rows(path: str | Path) -> list[list[float]]:
    """YOLO txt 한 파일을 숫자 row 리스트로 읽는다.

    반환 형식은 다음과 같다.
    - detection GT: `[class_id, xc, yc, w, h]`
    - detection prediction: `[class_id, xc, yc, w, h, score]`

    줄이 비어 있거나 형식이 너무 짧으면 조용히 건너뛴다. 예측 결과를 사람이
    수동 수정하는 과정에서 빈 줄이 끼는 경우가 종종 있기 때문이다.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            continue

        rows.append([float(part) for part in parts])
    return rows


def yolo_row_to_detection(row: list[float], width: int, height: int) -> tuple[int, list[float], float]:
    """YOLO 정규화 row를 `(class_id, xyxy_box, score)`로 바꾼다."""
    label = int(row[0])
    xc, yc, box_w, box_h = row[1:5]
    score = float(row[5]) if len(row) >= 6 else 1.0

    x1 = (xc - box_w / 2.0) * width
    y1 = (yc - box_h / 2.0) * height
    x2 = (xc + box_w / 2.0) * width
    y2 = (yc + box_h / 2.0) * height
    return label, [x1, y1, x2, y2], score


def normalize_class_name_map(class_names: dict[int, str] | list[str]) -> dict[int, str]:
    """클래스 이름 입력을 항상 `class_id -> class_name` dict로 통일한다."""
    if isinstance(class_names, dict):
        return {int(key): str(value) for key, value in sorted(class_names.items(), key=lambda item: int(item[0]))}
    return {idx: str(name) for idx, name in enumerate(class_names)}
