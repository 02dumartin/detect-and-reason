'''
YOLO 데이터셋 준비 스크립트
    raw 데이터셋이 없으면 다운로드하고, 있으면 재사용
    laboro / tomatod / merge 형태로 데이터셋 구성
    3클래스 데이터셋을 먼저 생성
    1클래스 데이터셋은 기존 3클래스 YOLO 결과를 복사한 뒤 모든 라벨을 tomato(0)로 변경
    validation은 test 이미지에서 20장을 선택해 복사
    merge는 laboro big/little의 train+test -> train, val -> val, tomatod는 train/val/test 유지
    YOLO 형식으로 저장 후 data.yaml 생성

출력 디렉토리:
    - data/yolo/laboro_big_3cls
    - data/yolo/laboro_big_1cls
    ...
    - data/coco/laboro_big_3cls
    - data/coco/laboro_big_1cls
    ...

사용법:
    python scripts/prepare_dataset.py --root laboro --dataset big --classes 3 --force
    python scripts/prepare_dataset.py --root laboro --dataset big --classes 3 --format coco --force
    python scripts/prepare_dataset.py --root merge --classes 3 --force
'''
import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from PIL import Image, ImageOps


RESIZE_MAX = (1333, 800)
VAL_SAMPLE_COUNT = 20
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
YOLO_DIR = DATA_DIR / "yolo"
RFDETR_DIR = DATA_DIR / "coco"
COCO_FIXED_DIR = RAW_DIR / "coco_fixed"

LABORO_URLS = {
    "big": "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_big.zip",
    "little": "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_little.zip",
}

LABORO_CLASS_MAP = {
    "fully_ripened": "fully ripened tomato",
    "half_ripened": "half ripened tomato",
    "green": "green tomato",
}

TOMATOD_TO_LABORO = {
    "fully-ripe": "fully ripened tomato",
    "semi-ripe": "half ripened tomato",
    "unripe": "green tomato",
}

THREE_CLASS_NAMES = [
    "fully ripened tomato",
    "half ripened tomato",
    "green tomato",
]


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱"""
    parser = argparse.ArgumentParser(description="YOLO dataset preparation entrypoint")
    parser.add_argument(
        "--root",
        type=str,
        choices=["laboro", "tomatod", "merge"],
        required=True,
        help="준비할 데이터셋 root",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["big", "little"],
        help="laboro에서 사용할 데이터셋",
    )
    parser.add_argument(
        "--classes",
        type=int,
        choices=[1, 3],
        required=True,
        help="클래스 수",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["yolo", "coco", "both"],
        default="both",
        help="export 대상: yolo만, coco만, 또는 둘 다 (기존 yolo 유지 시 --format coco 사용)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 존재하는 출력 디렉토리를 삭제하고 다시 생성",
    )
    return parser.parse_args()


def export_formats_arg(format_arg: str) -> tuple[bool, bool]:
    """CLI --format 값을 (export_yolo, export_rfdetr) 플래그로 변환"""
    if format_arg == "yolo":
        return True, False
    if format_arg == "coco":
        return False, True
    return True, True


def ensure_dirs() -> None:
    """데이터 준비에 필요한 기본 디렉토리를 생성합니다."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    RFDETR_DIR.mkdir(parents=True, exist_ok=True)
    COCO_FIXED_DIR.mkdir(parents=True, exist_ok=True)


def download_laboro_dataset(name: str) -> Path:
    """Laboro raw 데이터셋을 다운로드하고 압축 해제"""
    extract_dir = RAW_DIR / f"laboro_tomato_{name}"
    if extract_dir.exists():
        print(f"[skip] laboro {name} raw already exists: {extract_dir}")
        return extract_dir

    zip_path = RAW_DIR / f"laboro_tomato_{name}.zip"
    print(f"[download] laboro {name}: {LABORO_URLS[name]}")
    urllib.request.urlretrieve(LABORO_URLS[name], zip_path)
    print(f"[extract] {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()
    return extract_dir


def ensure_raw_dataset(root: str, dataset: str | None) -> None:
    """요청한 root에 필요한 raw 데이터셋이 있는지 확인하고, 없으면 다운로드"""
    if root == "laboro":
        if dataset is None:
            raise ValueError("--dataset is required when root=laboro")
        download_laboro_dataset(dataset)
        return

    if root == "merge":
        return

    tomatod_root = RAW_DIR / "tomatOD"
    if not tomatod_root.exists():
        raise FileNotFoundError(
            f"TomatOD raw dataset is missing: {tomatod_root}. "
            "Copy it into data/raw before preparing tomatod."
        )


def reset_output_dir(path: Path, force: bool) -> None:
    """출력 디렉토리를 초기화"""
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists. Use --force to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def coco_ensure_iscrowd(path: Path) -> Path:
    """COCO annotation에 iscrowd가 없으면 보정하고, segmentation을 제거한 JSON 파일을 반환"""
    dataset_name = path.parents[1].parent.name   # laboro_tomato_big / laboro_tomato_little / tomatOD
    split_name = path.stem                       # train / test
    fixed_path = COCO_FIXED_DIR / f"{dataset_name}_{split_name}_bbox_only.json"

    if fixed_path.exists():
        return fixed_path

    with path.open("r") as f:
        data = json.load(f)

    changed = False
    categories = data.get("categories", [])
    if categories and isinstance(categories[0].get("name"), list):
        names = categories[0]["name"]
        data["categories"] = [
            {
                "id": idx,
                "name": name,
                "supercategory": categories[0].get("supercategory", "tomato"),
            }
            for idx, name in enumerate(names, start=1)
        ]
        changed = True

    for ann in data.get("annotations", []):
        if "iscrowd" not in ann:
            ann["iscrowd"] = ann.get("is_crowd", 0)
            changed = True

        # segmentation 제거 - bbox만 사용
        if "segmentation" in ann:
            del ann["segmentation"]
            changed = True

    if not changed:
        return path

    with fixed_path.open("w") as f:
        json.dump(data, f)

    return fixed_path


def resize_keep_ratio(img_path: Path, max_size: tuple[int, int] = RESIZE_MAX) -> None:
    """이미지를 비율 유지 상태로 최대 크기 안에 들어오도록 축소"""
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        width, height = im.size
        scale = min(max_size[0] / width, max_size[1] / height)
        if scale >= 1:
            return
        new_width = int(width * scale)
        new_height = int(height * scale)
        im.resize((new_width, new_height), Image.Resampling.LANCZOS).save(img_path, quality=95)


def resize_split_images(output_dir: Path) -> None:
    """train / val / test / valid 이미지 전체를 리사이즈 (YOLO용; 정규화 라벨이라 좌표 보정 불필요)"""
    for split in ("train", "test", "val", "valid"):
        img_dir = output_dir / split / "images"
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.iterdir():
            if img_path.is_file():
                resize_keep_ratio(img_path)


def resize_coco_split_images(output_dir: Path, max_size: tuple[int, int] = RESIZE_MAX) -> None:
    """RF-DETR(COCO) split을 리사이즈하면서 annotation 좌표/area/이미지 크기도 함께 보정한다.

    COCO bbox는 절대 픽셀 좌표라 이미지만 줄이면 박스가 어긋난다(학습 불가).
    이미지 축소와 동시에 같은 scale로 bbox/area와 image width/height를 갱신한다.
    """
    for split in ("train", "test", "val", "valid"):
        split_dir = output_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        images_dir = split_dir / "images"
        if not ann_path.is_file() or not images_dir.is_dir():
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        for image in data.get("images", []):
            img_path = images_dir / Path(image["file_name"]).name
            if not img_path.is_file():
                continue

            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im)
                width, height = im.size
                scale = min(max_size[0] / width, max_size[1] / height)
                if scale >= 1:
                    continue
                new_width = int(width * scale)
                new_height = int(height * scale)
                im.resize((new_width, new_height), Image.Resampling.LANCZOS).save(img_path, quality=95)

            image["width"] = new_width
            image["height"] = new_height
            for ann in anns_by_image.get(int(image["id"]), []):
                x, y, box_w, box_h = ann["bbox"]
                nx = max(0.0, min(x * scale, new_width))
                ny = max(0.0, min(y * scale, new_height))
                nw = max(0.0, min(box_w * scale, new_width - nx))
                nh = max(0.0, min(box_h * scale, new_height - ny))
                ann["bbox"] = [nx, ny, nw, nh]
                ann["area"] = nw * nh

        ann_path.write_text(json.dumps(data), encoding="utf-8")


def write_rfdetr_split(ds: sv.DetectionDataset, split_dir: Path) -> None:
    """DetectionDataset을 RF-DETR Roboflow COCO split으로 저장"""
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(ds.classes)
    ]
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for image_id, (image_path, detection) in enumerate(ds.annotations.items(), start=1):
        src_path = Path(image_path)
        dest_path = images_dir / src_path.name
        shutil.copy2(src_path, dest_path)

        with Image.open(dest_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": src_path.name,
                "width": width,
                "height": height,
            }
        )

        if detection.xyxy is None:
            continue
        for index in range(len(detection)):
            x1, y1, x2, y2 = detection.xyxy[index]
            box_w = max(0.0, float(x2 - x1))
            box_h = max(0.0, float(y2 - y1))
            if box_w <= 0 or box_h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(detection.class_id[index]),
                    "bbox": [float(x1), float(y1), box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "info": {"description": "RF-DETR dataset"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ann_path = split_dir / "_annotations.coco.json"
    with ann_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)


def create_rfdetr_val_from_test(output_dir: Path) -> int:
    """test split에서 20장을 선택해 RF-DETR valid split을 생성"""
    valid_dir = output_dir / "valid"
    test_dir = output_dir / "test"
    valid_images_dir = valid_dir / "images"
    valid_images_dir.mkdir(parents=True, exist_ok=True)

    test_ann_path = test_dir / "_annotations.coco.json"
    with test_ann_path.open("r", encoding="utf-8") as handle:
        test_coco = json.load(handle)

    test_images = sorted(test_coco.get("images", []), key=lambda item: item["file_name"])
    rng = random.Random(RANDOM_SEED)
    samples = rng.sample(test_images, min(VAL_SAMPLE_COUNT, len(test_images)))
    sample_ids = {int(item["id"]) for item in samples}

    valid_images = []
    for image in test_coco.get("images", []):
        if int(image["id"]) not in sample_ids:
            continue
        image_name = Path(image["file_name"]).name
        shutil.copy2(test_dir / "images" / image_name, valid_images_dir / image_name)
        valid_images.append({**image, "file_name": image_name})

    valid_annotations = [
        ann for ann in test_coco.get("annotations", []) if int(ann["image_id"]) in sample_ids
    ]
    valid_coco = {
        "info": {"description": "RF-DETR validation split"},
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": test_coco.get("categories", []),
    }
    with (valid_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(valid_coco, handle)
    return len(valid_images)


def export_rfdetr_dataset(
    ds_train: sv.DetectionDataset,
    ds_test: sv.DetectionDataset,
    output_dir: Path,
    dataset_name: str,
    class_names: list[str],
    force: bool,
) -> None:
    """DetectionDataset을 RF-DETR Roboflow COCO 레이아웃으로 저장"""
    reset_output_dir(output_dir, force)
    write_rfdetr_split(ds_train, output_dir / "train")
    write_rfdetr_split(ds_test, output_dir / "test")
    create_rfdetr_val_from_test(output_dir)
    resize_coco_split_images(output_dir)
    print_summary(output_dir, dataset_name, class_names)


def create_val_from_test(output_dir: Path) -> int:
    """test 이미지에서 20장을 선택해 val로 복사"""
    val_images_dir = output_dir / "val" / "images"
    val_labels_dir = output_dir / "val" / "labels"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    test_images = sorted(
        path.name for path in (output_dir / "test" / "images").iterdir() if path.is_file()
    )
    rng = random.Random(RANDOM_SEED)
    samples = rng.sample(test_images, min(VAL_SAMPLE_COUNT, len(test_images)))

    for image_name in samples:
        label_name = f"{Path(image_name).stem}.txt"
        shutil.copy2(output_dir / "test" / "images" / image_name, val_images_dir / image_name)
        shutil.copy2(output_dir / "test" / "labels" / label_name, val_labels_dir / label_name)

    return len(samples)


def write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    """YOLO 학습용 data.yaml 파일을 생성"""
    yaml_path = output_dir / "data.yaml"
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=False, sort_keys=False)


def count_images(split_dir: Path) -> int:
    """split 디렉토리의 이미지 개수를 반환"""
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        return 0
    return sum(1 for path in images_dir.iterdir() if path.is_file())


def print_summary(output_dir: Path, dataset_name: str, class_names: list[str]) -> None:
    """생성된 데이터셋의 간략한 정보를 출력"""
    train_count = count_images(output_dir / "train")
    val_count = count_images(output_dir / "val")
    if val_count == 0:
        val_count = count_images(output_dir / "valid")
    test_count = count_images(output_dir / "test")
    print("=" * 50)
    print(f"saved: {output_dir}")
    print(f"데이터셋: {dataset_name}")
    print(f"클래스: {class_names}")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    print("=" * 50)


def remap_laboro_dataset(ds: sv.DetectionDataset, classes: int) -> None:
    """Laboro 클래스 체계를 3클래스 또는 1클래스로 재매핑"""
    def base_class(name: str) -> str:
        if name.startswith(("b_", "l_")):
            return name.split("_", 1)[1]
        return name

    if classes == 1:
        new_class_names = ["tomato"]
        old_to_new = {name: 0 for name in ds.classes}
    else:
        new_class_names = THREE_CLASS_NAMES
        old_to_new = {
            name: new_class_names.index(LABORO_CLASS_MAP[base_class(name)])
            for name in ds.classes
        }

    for _, det in ds.annotations.items():
        det.class_id = np.array([old_to_new[ds.classes[class_id]] for class_id in det.class_id])
    ds.classes = new_class_names


def remap_tomatod_dataset(ds: sv.DetectionDataset, source_classes: list[str], classes: int) -> None:
    """TomatOD 클래스 체계를 Laboro 기준 3클래스 또는 1클래스로 재매핑"""
    if classes == 1:
        for _, det in ds.annotations.items():
            det.class_id = np.zeros_like(det.class_id)
        ds.classes = ["tomato"]
        return

    old_to_new = {}
    for old_idx, class_name in enumerate(source_classes):
        mapped_name = TOMATOD_TO_LABORO.get(class_name)
        if mapped_name is not None:
            old_to_new[old_idx] = THREE_CLASS_NAMES.index(mapped_name)

    for _, det in ds.annotations.items():
        det.class_id = np.array([old_to_new[class_id] for class_id in det.class_id])
    ds.classes = THREE_CLASS_NAMES


def load_laboro_raw(dataset: str) -> tuple[sv.DetectionDataset, sv.DetectionDataset]:
    """Laboro raw COCO 데이터셋을 로드(segmentation 제거, bbox만 사용)."""
    base_path = RAW_DIR / f"laboro_tomato_{dataset}" / f"laboro_{dataset}"
    train_json = coco_ensure_iscrowd(base_path / "annotations" / "train.json")
    test_json = coco_ensure_iscrowd(base_path / "annotations" / "test.json")
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=str(base_path / "train"),
        annotations_path=str(train_json),
    )
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=str(base_path / "test"),
        annotations_path=str(test_json),
    )
    return ds_train, ds_test


def load_tomatod_raw() -> tuple[sv.DetectionDataset, sv.DetectionDataset]:
    """TomatOD raw COCO 데이터셋을 로드"""
    train_json = coco_ensure_iscrowd(RAW_DIR / "tomatOD" / "annotations" / "train.json")
    test_json = coco_ensure_iscrowd(RAW_DIR / "tomatOD" / "annotations" / "test.json")
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=str(RAW_DIR / "tomatOD" / "train"),
        annotations_path=str(train_json),
    )
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=str(RAW_DIR / "tomatOD" / "test"),
        annotations_path=str(test_json),
    )
    return ds_train, ds_test


def export_paired_datasets(
    ds_train: sv.DetectionDataset,
    ds_test: sv.DetectionDataset,
    yolo_output_dir: Path,
    rfdetr_output_dir: Path,
    dataset_name: str,
    class_names: list[str],
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> None:
    """YOLO와/또는 RF-DETR 데이터셋을 같은 DetectionDataset에서 export"""
    if export_yolo:
        export_detection_dataset(
            ds_train, ds_test, yolo_output_dir, dataset_name, class_names, force
        )
    if export_rfdetr:
        export_rfdetr_dataset(
            ds_train, ds_test, rfdetr_output_dir, dataset_name, class_names, force
        )


def export_detection_dataset(
    ds_train: sv.DetectionDataset,
    ds_test: sv.DetectionDataset,
    output_dir: Path,
    dataset_name: str,
    class_names: list[str],
    force: bool,
) -> None:
    """DetectionDataset을 YOLO 형식으로 저장하고 val, resize, yaml 생성을 수행"""
    reset_output_dir(output_dir, force)
    ds_train.as_yolo(
        images_directory_path=str(output_dir / "train" / "images"),
        annotations_directory_path=str(output_dir / "train" / "labels"),
    )
    ds_test.as_yolo(
        images_directory_path=str(output_dir / "test" / "images"),
        annotations_directory_path=str(output_dir / "test" / "labels"),
    )
    create_val_from_test(output_dir)
    resize_split_images(output_dir)
    write_data_yaml(output_dir, class_names)
    print_summary(output_dir, dataset_name, class_names)


def output_name_for(root: str, dataset: str | None, classes: int) -> str:
    """출력 디렉토리 이름을 생성"""
    if root == "laboro":
        if dataset is None:
            raise ValueError("dataset is required for laboro outputs")
        return f"laboro_{dataset}_{classes}cls"
    if root == "tomatod":
        return f"tomatod_{classes}cls"
    if root == "merge":
        return f"merge_{classes}cls"
    raise ValueError(f"unsupported root: {root}")


def export_laboro_3cls(
    dataset: str,
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """Laboro 3클래스 데이터셋을 생성"""
    ds_train, ds_test = load_laboro_raw(dataset)
    remap_laboro_dataset(ds_train, 3)
    remap_laboro_dataset(ds_test, 3)
    yolo_output_dir = YOLO_DIR / output_name_for("laboro", dataset, 3)
    rfdetr_output_dir = RFDETR_DIR / output_name_for("laboro", dataset, 3)
    export_paired_datasets(
        ds_train,
        ds_test,
        yolo_output_dir,
        rfdetr_output_dir,
        dataset,
        THREE_CLASS_NAMES,
        force,
        export_yolo=export_yolo,
        export_rfdetr=export_rfdetr,
    )
    return rfdetr_output_dir if export_rfdetr and not export_yolo else yolo_output_dir


def prepare_laboro_3cls(
    dataset: str,
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """Laboro 3클래스 결과를 최종 출력 경로에 저장"""
    return export_laboro_3cls(
        dataset,
        force,
        export_yolo=export_yolo,
        export_rfdetr=export_rfdetr,
    )


def export_tomatod_3cls(
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """TomatOD 3클래스 데이터셋을 생성"""
    ds_train, ds_test = load_tomatod_raw()
    source_classes = list(ds_train.classes)
    remap_tomatod_dataset(ds_train, source_classes, 3)
    remap_tomatod_dataset(ds_test, source_classes, 3)
    yolo_output_dir = YOLO_DIR / output_name_for("tomatod", None, 3)
    rfdetr_output_dir = RFDETR_DIR / output_name_for("tomatod", None, 3)
    export_paired_datasets(
        ds_train,
        ds_test,
        yolo_output_dir,
        rfdetr_output_dir,
        "tomatod",
        THREE_CLASS_NAMES,
        force,
        export_yolo=export_yolo,
        export_rfdetr=export_rfdetr,
    )
    return rfdetr_output_dir if export_rfdetr and not export_yolo else yolo_output_dir


def prepare_tomatod_3cls(
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """TomatOD 3클래스 결과를 최종 출력 경로에 저장"""
    return export_tomatod_3cls(
        force,
        export_yolo=export_yolo,
        export_rfdetr=export_rfdetr,
    )


def convert_yolo_3cls_to_1cls(
    src_dir: Path,
    dst_dir: Path,
    dataset_name: str,
    force: bool,
) -> Path:
    """기존 3클래스 YOLO 데이터셋을 복사한 뒤 모든 라벨을 1클래스로 변환"""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"3cls dataset is required first: {src_dir}")

    reset_output_dir(dst_dir, force)
    for split in ("train", "test", "val"):
        for subdir in ("images", "labels"):
            (dst_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    for split in ("train", "test", "val"):
        src_images_dir = src_dir / split / "images"
        dst_images_dir = dst_dir / split / "images"
        if src_images_dir.is_dir():
            for image_path in src_images_dir.iterdir():
                if image_path.is_file():
                    shutil.copy2(image_path, dst_images_dir / image_path.name)

        src_labels_dir = src_dir / split / "labels"
        dst_labels_dir = dst_dir / split / "labels"
        if src_labels_dir.is_dir():
            for label_path in src_labels_dir.iterdir():
                if not label_path.is_file():
                    continue
                with label_path.open("r") as f:
                    lines = f.readlines()
                remapped_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) < 5:
                        continue
                    parts[0] = "0"
                    remapped_lines.append(" ".join(parts) + "\n")
                with (dst_labels_dir / label_path.name).open("w") as f:
                    f.writelines(remapped_lines)

    write_data_yaml(dst_dir, ["tomato"])
    print_summary(dst_dir, dataset_name, ["tomato"])
    return dst_dir


def convert_rfdetr_3cls_to_1cls(
    src_dir: Path,
    dst_dir: Path,
    dataset_name: str,
    force: bool,
) -> Path:
    """기존 3클래스 RF-DETR COCO 데이터셋을 1클래스로 변환"""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"3cls RF-DETR dataset is required first: {src_dir}")

    reset_output_dir(dst_dir, force)
    for split in ("train", "test", "valid"):
        src_split_dir = src_dir / split
        dst_split_dir = dst_dir / split
        ann_path = src_split_dir / "_annotations.coco.json"
        if not ann_path.is_file():
            continue

        dst_images_dir = dst_split_dir / "images"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        with ann_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        for image in coco.get("images", []):
            image_name = Path(image["file_name"]).name
            shutil.copy2(src_split_dir / "images" / image_name, dst_images_dir / image_name)
            image["file_name"] = image_name

        for ann in coco.get("annotations", []):
            ann["category_id"] = 0
        coco["categories"] = [{"id": 0, "name": "tomato", "supercategory": "object"}]

        with (dst_split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
            json.dump(coco, handle)

    print_summary(dst_dir, dataset_name, ["tomato"])
    return dst_dir


def _normalize_rfdetr_split_name(split: str) -> str:
    return "valid" if split == "val" else split


def _merge_rfdetr_split(
    *,
    parts: list[tuple[Path, str, str]],
    dst_split: str,
    dst_dir: Path,
    next_image_id: int,
    next_ann_id: int,
) -> tuple[int, int]:
    dst_split_dir = dst_dir / dst_split
    dst_images_dir = dst_split_dir / "images"
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []

    for src_dir, src_split, mapped_dst_split in parts:
        if mapped_dst_split != dst_split:
            continue
        src_split_name = _normalize_rfdetr_split_name(src_split)
        src_split_dir = src_dir / src_split_name
        ann_path = src_split_dir / "_annotations.coco.json"
        if not ann_path.is_file():
            continue

        with ann_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        image_id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            old_id = int(image["id"])
            image_name = Path(image["file_name"]).name
            src_image_path = src_split_dir / "images" / image_name
            if not src_image_path.is_file():
                continue
            shutil.copy2(src_image_path, dst_images_dir / image_name)
            image_id_map[old_id] = next_image_id
            merged_images.append(
                {
                    "id": next_image_id,
                    "file_name": image_name,
                    "width": image["width"],
                    "height": image["height"],
                }
            )
            next_image_id += 1

        for ann in coco.get("annotations", []):
            old_image_id = int(ann["image_id"])
            if old_image_id not in image_id_map:
                continue
            merged_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id_map[old_image_id],
                    "category_id": int(ann["category_id"]),
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann_id += 1

    merged_coco = {
        "info": {"description": f"RF-DETR merged {dst_split}"},
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(THREE_CLASS_NAMES)
        ],
    }
    with (dst_split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_coco, handle)
    return next_image_id, next_ann_id


def prepare_merge_rfdetr_3cls(force: bool) -> Path:
    """기존 RF-DETR 3cls 결과를 merge RF-DETR 데이터셋으로 합친다."""
    laboro_big_src = RFDETR_DIR / output_name_for("laboro", "big", 3)
    laboro_little_src = RFDETR_DIR / output_name_for("laboro", "little", 3)
    tomatod_src = RFDETR_DIR / output_name_for("tomatod", None, 3)
    for src_dir in (laboro_big_src, laboro_little_src, tomatod_src):
        if not src_dir.is_dir():
            raise FileNotFoundError(f"merge RF-DETR requires existing dataset first: {src_dir}")

    output_dir = RFDETR_DIR / output_name_for("merge", None, 3)
    reset_output_dir(output_dir, force)

    parts = [
        (laboro_big_src, "train", "train"),
        (laboro_big_src, "test", "train"),
        (laboro_big_src, "valid", "valid"),
        (laboro_little_src, "train", "train"),
        (laboro_little_src, "test", "train"),
        (laboro_little_src, "valid", "valid"),
        (tomatod_src, "train", "train"),
        (tomatod_src, "valid", "valid"),
        (tomatod_src, "test", "test"),
    ]

    next_image_id = 1
    next_ann_id = 1
    for split in ("train", "valid", "test"):
        next_image_id, next_ann_id = _merge_rfdetr_split(
            parts=parts,
            dst_split=split,
            dst_dir=output_dir,
            next_image_id=next_image_id,
            next_ann_id=next_ann_id,
        )

    resize_coco_split_images(output_dir)
    print_summary(output_dir, "merge", THREE_CLASS_NAMES)
    return output_dir


def copy_yolo_split(src_dir: Path, dst_dir: Path, src_split: str, dst_split: str) -> None:
    """YOLO split 디렉토리를 다른 split으로 복사"""
    for subdir in ("images", "labels"):
        source = src_dir / src_split / subdir
        target = dst_dir / dst_split / subdir
        if not source.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)


def prepare_merge_3cls(
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """기존 laboro big/little, tomatod 결과를 합쳐 merge 3클래스 데이터셋을 생성"""
    output_dir = YOLO_DIR / output_name_for("merge", None, 3)

    if export_yolo:
        laboro_big_src = YOLO_DIR / output_name_for("laboro", "big", 3)
        laboro_little_src = YOLO_DIR / output_name_for("laboro", "little", 3)
        tomatod_src = YOLO_DIR / output_name_for("tomatod", None, 3)

        for src_dir in (laboro_big_src, laboro_little_src, tomatod_src):
            if not src_dir.is_dir():
                raise FileNotFoundError(f"merge requires existing dataset first: {src_dir}")

        reset_output_dir(output_dir, force)
        for split in ("train", "test", "val"):
            for subdir in ("images", "labels"):
                (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

        for laboro_src in (laboro_big_src, laboro_little_src):
            copy_yolo_split(laboro_src, output_dir, "train", "train")
            copy_yolo_split(laboro_src, output_dir, "test", "train")
            copy_yolo_split(laboro_src, output_dir, "val", "val")

        for split in ("train", "test", "val"):
            copy_yolo_split(tomatod_src, output_dir, split, split)

        write_data_yaml(output_dir, THREE_CLASS_NAMES)
        print_summary(output_dir, "merge", THREE_CLASS_NAMES)

    if export_rfdetr:
        prepare_merge_rfdetr_3cls(force)

    if export_rfdetr and not export_yolo:
        return RFDETR_DIR / output_name_for("merge", None, 3)
    return output_dir


def prepare_laboro(
    classes: int,
    dataset: str,
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """Laboro 데이터셋을 클래스 수에 맞게 준비"""
    if classes == 3:
        return prepare_laboro_3cls(
            dataset,
            force,
            export_yolo=export_yolo,
            export_rfdetr=export_rfdetr,
        )

    last_output = None
    if export_yolo:
        src_dir = YOLO_DIR / output_name_for("laboro", dataset, 3)
        dst_dir = YOLO_DIR / output_name_for("laboro", dataset, 1)
        last_output = convert_yolo_3cls_to_1cls(src_dir, dst_dir, dataset, force)
    if export_rfdetr:
        src_rfdetr = RFDETR_DIR / output_name_for("laboro", dataset, 3)
        dst_rfdetr = RFDETR_DIR / output_name_for("laboro", dataset, 1)
        last_output = convert_rfdetr_3cls_to_1cls(src_rfdetr, dst_rfdetr, dataset, force)
    if last_output is None:
        raise ValueError("At least one of export_yolo or export_rfdetr must be enabled")
    return last_output


def prepare_tomatod(
    classes: int,
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """TomatOD 데이터셋을 클래스 수에 맞게 준비"""
    if classes == 3:
        return prepare_tomatod_3cls(
            force,
            export_yolo=export_yolo,
            export_rfdetr=export_rfdetr,
        )

    last_output = None
    if export_yolo:
        src_dir = YOLO_DIR / output_name_for("tomatod", None, 3)
        dst_dir = YOLO_DIR / output_name_for("tomatod", None, 1)
        last_output = convert_yolo_3cls_to_1cls(src_dir, dst_dir, "tomatod", force)
    if export_rfdetr:
        src_rfdetr = RFDETR_DIR / output_name_for("tomatod", None, 3)
        dst_rfdetr = RFDETR_DIR / output_name_for("tomatod", None, 1)
        last_output = convert_rfdetr_3cls_to_1cls(src_rfdetr, dst_rfdetr, "tomatod", force)
    if last_output is None:
        raise ValueError("At least one of export_yolo or export_rfdetr must be enabled")
    return last_output


def prepare_merge(
    classes: int,
    force: bool,
    *,
    export_yolo: bool = True,
    export_rfdetr: bool = True,
) -> Path:
    """Merge 데이터셋을 클래스 수에 맞게 준비"""
    if classes == 3:
        return prepare_merge_3cls(
            force,
            export_yolo=export_yolo,
            export_rfdetr=export_rfdetr,
        )

    last_output = None
    if export_yolo:
        src_dir = YOLO_DIR / output_name_for("merge", None, 3)
        dst_dir = YOLO_DIR / output_name_for("merge", None, 1)
        last_output = convert_yolo_3cls_to_1cls(src_dir, dst_dir, "merge", force)
    if export_rfdetr:
        src_rfdetr = RFDETR_DIR / output_name_for("merge", None, 3)
        dst_rfdetr = RFDETR_DIR / output_name_for("merge", None, 1)
        last_output = convert_rfdetr_3cls_to_1cls(src_rfdetr, dst_rfdetr, "merge", force)
    if last_output is None:
        raise ValueError("At least one of export_yolo or export_rfdetr must be enabled")
    return last_output


def main() -> None:
    """전체 데이터셋 준비 과정을 실행"""
    args = parse_args()
    ensure_dirs()

    if args.root == "laboro" and args.dataset is None:
        raise SystemExit("--dataset must be provided for root=laboro")
    if args.root != "laboro" and args.dataset is not None:
        raise SystemExit("--dataset is only valid for root=laboro")

    ensure_raw_dataset(args.root, args.dataset)
    export_yolo, export_rfdetr = export_formats_arg(args.format)

    if args.root == "laboro":
        prepare_laboro(
            args.classes,
            args.dataset,
            args.force,
            export_yolo=export_yolo,
            export_rfdetr=export_rfdetr,
        )
        return
    if args.root == "tomatod":
        prepare_tomatod(
            args.classes,
            args.force,
            export_yolo=export_yolo,
            export_rfdetr=export_rfdetr,
        )
        return
    prepare_merge(
        args.classes,
        args.force,
        export_yolo=export_yolo,
        export_rfdetr=export_rfdetr,
    )


if __name__ == "__main__":
    main()
