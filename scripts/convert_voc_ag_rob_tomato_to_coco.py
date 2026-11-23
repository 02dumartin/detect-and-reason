"""AgRobTomato VOC 데이터를 COCO 포맷(train/valid/test + _annotations.coco.json)으로 변환한다.

convert_voc_ag_rob_tomato_to_yolo.py와 동일한 split config(seed/ratios/class_mapping)를
공유하므로, 두 스크립트를 같은 split-config로 실행하면 YOLO/COCO 버전이 동일한
이미지 split을 갖는다 (dino/rf_detr 용).

Example:
    python scripts/convert_voc_ag_rob_tomato_to_coco.py \
        --split-config config/dataset/ag_rob_tomato_split.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import autorootcwd
import yaml

from src.config_loader import get_project_root

from convert_voc_ag_rob_tomato_to_yolo import build_source_to_idx, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgRobTomato VOC annotations to COCO format.")
    parser.add_argument("--split-config", type=str, required=True, help="Split/변환 설정 YAML 경로.")
    return parser.parse_args()


def build_coco_dict(
    image_filenames: list[str],
    voc_root: Path,
    source_to_idx: dict[str, int],
    class_names: dict[int, str],
) -> dict:
    """한 split의 이미지 목록으로 COCO annotation dict를 만든다."""
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for image_id, image_filename in enumerate(image_filenames, start=1):
        xml_path = voc_root / "Annotations" / f"{Path(image_filename).stem}.xml"
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        images.append(
            {
                "id": image_id,
                "file_name": f"images/{image_filename}",
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in source_to_idx:
                continue
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
            w = xmax - xmin
            h = ymax - ymin

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": source_to_idx[name],
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": idx, "name": name, "supercategory": "tomato"} for idx, name in sorted(class_names.items())
    ]

    return {
        "info": {"description": "AgRobTomato (converted from VOC)"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write_split(
    split_name: str,
    coco_split_name: str,
    image_filenames: list[str],
    voc_root: Path,
    source_to_idx: dict[str, int],
    class_names: dict[int, str],
    output_root: Path,
) -> None:
    """한 split의 이미지를 복사하고 COCO annotation json을 저장한다."""
    split_dir = output_root / coco_split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for image_filename in image_filenames:
        shutil.copy2(voc_root / "JPEGImages" / image_filename, images_dir / image_filename)

    coco_dict = build_coco_dict(image_filenames, voc_root, source_to_idx, class_names)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco_dict))


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    cfg = yaml.safe_load(Path(args.split_config).read_text())

    voc_root = project_root / cfg["source"]["voc_root"]
    output_root = project_root / cfg["output"]["coco_root"]
    class_names = cfg["output"]["class_names"]

    source_to_idx = build_source_to_idx(cfg["source"]["class_mapping"], class_names)

    image_filenames = sorted(p.name for p in (voc_root / "JPEGImages").glob("*.jpg"))
    splits = split_dataset(image_filenames, cfg["ratios"], cfg["seed"])

    # rf_detr/dino 컨벤션: val split 디렉토리명은 "valid"
    coco_split_names = {"train": "train", "val": "valid", "test": "test"}

    for split_name, filenames in splits.items():
        write_split(split_name, coco_split_names[split_name], filenames, voc_root, source_to_idx, class_names, output_root)

    for split_name, filenames in splits.items():
        print(f"{split_name}: {len(filenames)} images")
    print(f"COCO data written to {output_root}")


if __name__ == "__main__":
    main()
