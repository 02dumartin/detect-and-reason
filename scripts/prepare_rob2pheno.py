"""
Prepare Rob2Pheno exports for detect-and-reason.

Source dataset:
    data/raw/Rob2Pheno/
      RGB_png/
      train_2class.JSON
      val_2class.JSON

Split rules:
    - train = train_2class.JSON
    - test = val_2class.JSON
    - val = random 20-image subset of test

Outputs:
    data/coco/Rob2Pheno_{1,2}cls/
    data/yolo/Rob2Pheno_{1,2}cls/

Notes:
    - Images are exported from RGB_png as .png files.
    - 2cls maps redfruit -> ripe(0), greenfruit -> unripe(1).
    - 1cls maps every annotation to tomato(0).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = BASE_DIR / "data" / "raw" / "Rob2Pheno"
SOURCE_IMAGE_DIRNAME = "RGB_png"
TRAIN_SPLIT_FILE = "train_2class.JSON"
TEST_SPLIT_FILE = "val_2class.JSON"
RF_SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}
YOLO_SPLITS = ("train", "val", "test")
CLASS_NAMES_BY_COUNT = {1: ["tomato"], 2: ["ripe", "unripe"]}
TWO_CLASS_NAME_MAP = {
    "redfruit": 0,
    "ripe": 0,
    "red": 0,
    "greenfruit": 1,
    "unripe": 1,
    "green": 1,
}
VAL_SAMPLE_SIZE = 20
VAL_SAMPLE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Rob2Pheno dataset exports")
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Rob2Pheno source root",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
        help="Class counts to export",
    )
    parser.add_argument(
        "--format",
        choices=["yolo", "coco", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to place image files in output directories",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs")
    parser.add_argument(
        "--val-sample-size",
        type=int,
        default=VAL_SAMPLE_SIZE,
        help="Number of test images to sample into val",
    )
    parser.add_argument(
        "--val-sample-seed",
        type=int,
        default=VAL_SAMPLE_SEED,
        help="Random seed for val sampling",
    )
    return parser.parse_args()


def coco_out_dir(num_classes: int) -> Path:
    return BASE_DIR / "data" / "coco" / f"Rob2Pheno_{num_classes}cls"


def yolo_out_dir(num_classes: int) -> Path:
    return BASE_DIR / "data" / "yolo" / f"Rob2Pheno_{num_classes}cls"


def _ensure_source_exists(source_root: Path) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Rob2Pheno source root not found: {source_root}")
    image_root = source_root / SOURCE_IMAGE_DIRNAME
    if not image_root.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {image_root}")
    for file_name in (TRAIN_SPLIT_FILE, TEST_SPLIT_FILE):
        split_path = source_root / file_name
        if not split_path.is_file():
            raise FileNotFoundError(f"Split JSON not found: {split_path}")


def _place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _normalize_source_category_map(coco_data: dict) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for category in coco_data.get("categories", []):
        old_id = int(category["id"])
        raw_name = str(category.get("name", "")).strip().lower().replace(" ", "")
        if raw_name not in TWO_CLASS_NAME_MAP:
            raise ValueError(f"Unsupported Rob2Pheno category name: {category.get('name')}")
        mapping[old_id] = TWO_CLASS_NAME_MAP[raw_name]
    if not mapping:
        raise ValueError("No categories found in Rob2Pheno source JSON")
    return mapping


def _target_image_name(source_file_name: str) -> str:
    return f"{Path(source_file_name).stem}.png"


def _target_image_path(source_root: Path, source_file_name: str) -> Path:
    return source_root / SOURCE_IMAGE_DIRNAME / _target_image_name(source_file_name)


def _load_coco_split(source_root: Path, file_name: str) -> dict:
    split_path = source_root / file_name
    return json.loads(split_path.read_text(encoding="utf-8"))


def _subset_coco_data(source_root: Path, source_data: dict, image_ids: set[int], *, split_name: str, num_classes: int) -> dict:
    category_map = _normalize_source_category_map(source_data)

    images = []
    source_images = sorted(source_data.get("images", []), key=lambda image: int(image["id"]))
    for image in source_images:
        image_id = int(image["id"])
        if image_id not in image_ids:
            continue
        target_file_name = _target_image_name(str(image["file_name"]))
        src_image = _target_image_path(source_root, str(image["file_name"]))
        if not src_image.is_file():
            raise FileNotFoundError(f"Source PNG image not found: {src_image}")
        images.append(
            {
                "id": image_id,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "file_name": target_file_name,
            }
        )

    annotations = []
    source_annotations = sorted(source_data.get("annotations", []), key=lambda ann: int(ann["id"]))
    for ann in source_annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_ids:
            continue
        if num_classes == 1:
            category_id = 0
        else:
            category_id = category_map[int(ann["category_id"])]
        new_ann = {
            "id": int(ann["id"]),
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(value) for value in ann.get("bbox", [])[:4]],
            "area": float(ann.get("area", 0.0)),
            "iscrowd": int(ann.get("iscrowd", 0)),
        }
        if "segmentation" in ann:
            new_ann["segmentation"] = ann["segmentation"]
        annotations.append(new_ann)

    categories = [
        {"id": class_id, "name": class_name, "supercategory": "tomato"}
        for class_id, class_name in enumerate(CLASS_NAMES_BY_COUNT[num_classes])
    ]
    return {
        "info": {"description": f"Rob2Pheno {num_classes}cls {split_name}"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _build_split_data(
    source_root: Path,
    *,
    num_classes: int,
    val_sample_size: int,
    val_sample_seed: int,
) -> dict[str, dict]:
    train_source = _load_coco_split(source_root, TRAIN_SPLIT_FILE)
    test_source = _load_coco_split(source_root, TEST_SPLIT_FILE)

    train_image_ids = {int(image["id"]) for image in train_source.get("images", [])}
    test_images = sorted(test_source.get("images", []), key=lambda image: int(image["id"]))
    test_image_ids = [int(image["id"]) for image in test_images]

    if val_sample_size > len(test_image_ids):
        raise ValueError(
            f"val sample size {val_sample_size} exceeds available test images {len(test_image_ids)}"
        )

    sampled_val_ids = set(random.Random(val_sample_seed).sample(test_image_ids, val_sample_size))
    test_image_id_set = set(test_image_ids)

    return {
        "train": _subset_coco_data(
            source_root,
            train_source,
            train_image_ids,
            split_name="train",
            num_classes=num_classes,
        ),
        "val": _subset_coco_data(
            source_root,
            test_source,
            sampled_val_ids,
            split_name="val",
            num_classes=num_classes,
        ),
        "test": _subset_coco_data(
            source_root,
            test_source,
            test_image_id_set,
            split_name="test",
            num_classes=num_classes,
        ),
    }


def _bbox_to_yolo_line(bbox: list[float], *, width: int, height: int, class_id: int) -> str:
    x, y, w, h = bbox
    x_center = (x + (w / 2.0)) / width
    y_center = (y + (h / 2.0)) / height
    norm_w = w / width
    norm_h = h / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def _write_data_yaml(out_dir: Path, num_classes: int) -> None:
    lines = [
        f"path: {out_dir.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES_BY_COUNT[num_classes])}",
        "names:",
    ]
    for class_id, class_name in enumerate(CLASS_NAMES_BY_COUNT[num_classes]):
        lines.append(f"  {class_id}: {class_name}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_coco(*, source_root: Path, split_data: dict[str, dict], num_classes: int, force: bool, link_mode: str) -> Path:
    out_dir = coco_out_dir(num_classes)
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} already exists. Use --force to rebuild.")
        shutil.rmtree(out_dir)

    for split_name, split_payload in split_data.items():
        split_root = out_dir / RF_SPLIT_NAMES[split_name]
        images_dir = split_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for image in split_payload["images"]:
            source_path = source_root / SOURCE_IMAGE_DIRNAME / image["file_name"]
            _place_file(source_path, images_dir / image["file_name"], link_mode)
        ann_path = split_root / "_annotations.coco.json"
        ann_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rob2pheno] COCO {num_classes}cls -> {out_dir}")
    return out_dir


def export_yolo(*, source_root: Path, split_data: dict[str, dict], num_classes: int, force: bool, link_mode: str) -> Path:
    out_dir = yolo_out_dir(num_classes)
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} already exists. Use --force to rebuild.")
        shutil.rmtree(out_dir)

    for split_name in YOLO_SPLITS:
        split_root = out_dir / split_name
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        payload = split_data[split_name]
        images_by_id = {int(image["id"]): image for image in payload["images"]}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in payload["annotations"]:
            anns_by_image[int(ann["image_id"])].append(ann)

        for image in payload["images"]:
            image_id = int(image["id"])
            file_name = str(image["file_name"])
            source_path = source_root / SOURCE_IMAGE_DIRNAME / file_name
            _place_file(source_path, images_dir / file_name, link_mode)

            label_lines = []
            for ann in anns_by_image.get(image_id, []):
                label_lines.append(
                    _bbox_to_yolo_line(
                        ann["bbox"],
                        width=int(images_by_id[image_id]["width"]),
                        height=int(images_by_id[image_id]["height"]),
                        class_id=int(ann["category_id"]),
                    )
                )
            label_path = labels_dir / f"{Path(file_name).stem}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    _write_data_yaml(out_dir, num_classes)
    print(f"[rob2pheno] YOLO {num_classes}cls -> {out_dir}")
    return out_dir


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    _ensure_source_exists(source_root)

    export_coco_flag = args.format in {"coco", "both"}
    export_yolo_flag = args.format in {"yolo", "both"}
    class_counts = sorted(set(args.classes))

    for num_classes in class_counts:
        split_payloads = _build_split_data(
            source_root,
            num_classes=num_classes,
            val_sample_size=args.val_sample_size,
            val_sample_seed=args.val_sample_seed,
        )
        for split_name, payload in split_payloads.items():
            print(
                f"[rob2pheno] {num_classes}cls {split_name}: "
                f"{len(payload['images'])} images, {len(payload['annotations'])} annotations"
            )
        if export_coco_flag:
            export_coco(
                source_root=source_root,
                split_data=split_payloads,
                num_classes=num_classes,
                force=args.force,
                link_mode=args.link_mode,
            )
        if export_yolo_flag:
            export_yolo(
                source_root=source_root,
                split_data=split_payloads,
                num_classes=num_classes,
                force=args.force,
                link_mode=args.link_mode,
            )

    print("[rob2pheno] Done.")


if __name__ == "__main__":
    main()
