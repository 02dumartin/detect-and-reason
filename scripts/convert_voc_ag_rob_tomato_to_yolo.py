"""AgRobTomato VOC 데이터를 YOLO 포맷(train/val/test + data.yaml)으로 변환한다.

Example:
    python scripts/convert_voc_ag_rob_tomato_to_yolo.py \
        --split-config config/dataset/ag_rob_tomato_split.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import autorootcwd
import yaml

from src.config_loader import get_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgRobTomato VOC annotations to YOLO format.")
    parser.add_argument("--split-config", type=str, required=True, help="Split/변환 설정 YAML 경로.")
    return parser.parse_args()


def build_source_to_idx(class_mapping: dict[str, str], target_class_names: dict[int, str]) -> dict[str, int]:
    """VOC 원본 라벨명을 target class_names의 인덱스로 바로 매핑하는 테이블을 만든다."""
    target_name_to_idx = {name: idx for idx, name in target_class_names.items()}
    return {source_name: target_name_to_idx[target_name] for source_name, target_name in class_mapping.items()}


def voc_box_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> tuple[float, float, float, float]:
    """VOC 절대좌표 bbox를 YOLO 정규화 (cx, cy, w, h)로 변환한다."""
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return cx, cy, w, h


def convert_annotation(xml_path: Path, source_to_idx: dict[str, int]) -> tuple[str, list[str]] | None:
    """annotation XML 하나를 (이미지 파일명, YOLO 라벨 라인 목록)으로 변환한다."""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in source_to_idx:
            continue
        box = obj.find("bndbox")
        cx, cy, w, h = voc_box_to_yolo(
            float(box.findtext("xmin")),
            float(box.findtext("ymin")),
            float(box.findtext("xmax")),
            float(box.findtext("ymax")),
            width,
            height,
        )
        lines.append(f"{source_to_idx[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return filename, lines


def split_dataset(filenames: list[str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    """파일명 목록을 셔플한 뒤 비율대로 train/val/test로 나눈다."""
    shuffled = filenames.copy()
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_split(
    split_name: str,
    image_filenames: list[str],
    voc_root: Path,
    source_to_idx: dict[str, int],
    output_root: Path,
) -> None:
    """한 split(train/val/test)의 이미지/라벨을 YOLO 디렉토리 구조로 복사/생성한다."""
    images_dir = output_root / split_name / "images"
    labels_dir = output_root / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_filename in image_filenames:
        xml_path = voc_root / "Annotations" / f"{Path(image_filename).stem}.xml"
        _, lines = convert_annotation(xml_path, source_to_idx)

        shutil.copy2(voc_root / "JPEGImages" / image_filename, images_dir / image_filename)
        label_path = labels_dir / f"{Path(image_filename).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(output_root: Path, class_names: dict[int, str]) -> None:
    """ultralytics가 요구하는 data.yaml을 생성한다."""
    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    cfg = yaml.safe_load(Path(args.split_config).read_text())

    voc_root = project_root / cfg["source"]["voc_root"]
    output_root = project_root / cfg["output"]["yolo_root"]
    class_names = cfg["output"]["class_names"]

    source_to_idx = build_source_to_idx(cfg["source"]["class_mapping"], class_names)

    image_filenames = sorted(p.name for p in (voc_root / "JPEGImages").glob("*.jpg"))
    splits = split_dataset(image_filenames, cfg["ratios"], cfg["seed"])

    for split_name, filenames in splits.items():
        write_split(split_name, filenames, voc_root, source_to_idx, output_root)

    write_data_yaml(output_root, class_names)

    for split_name, filenames in splits.items():
        print(f"{split_name}: {len(filenames)} images")
    print(f"classes: {class_names}")
    print(f"data.yaml written to {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
