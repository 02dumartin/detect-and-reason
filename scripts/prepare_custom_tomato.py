"""
Custom tomato 데이터를 detect-and-reason 프로젝트에서 사용할 수 있게 준비한다.

원본 위치 (symlink):
    data/custom_tomato_dataset/
      custom_tomato_data_yolo/   ← YOLO (images/labels split 완료, 라벨 0/1=ripe/unripe)
      custom_tomato_data_coco/   ← flat COCO (_annotations.coco.json + images/)

생성물 (--classes 에 따라):
    data/yolo/custom_tomato_{N}cls/   ← 2cls=외부 wrapper, 1cls=라벨 0으로 합친 실체 복사본
    data/coco/custom_tomato_{N}cls/   ← RF-DETR Roboflow COCO layout

Usage:
    python scripts/prepare_custom_tomato.py --classes 2 --format both --force
    python scripts/prepare_custom_tomato.py --classes 1 --format coco --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE_DIR / "data" / "custom_tomato_dataset"
COCO_ROOT = DATASET_ROOT / "custom_tomato_data_coco"
YOLO_ROOT = DATASET_ROOT / "custom_tomato_data_yolo"

# 클래스 수별 이름. 1cls는 ripe/unripe를 모두 tomato(0)으로 합친다.
CLASS_NAMES_BY_N = {1: ["tomato"], 2: ["ripe", "unripe"]}
SPLITS = ("train", "val", "test")
RF_SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}


def yolo_out_dir(num_classes: int) -> Path:
    return BASE_DIR / "data" / "yolo" / f"custom_tomato_{num_classes}cls"


def coco_out_dir(num_classes: int) -> Path:
    return BASE_DIR / "data" / "coco" / f"custom_tomato_{num_classes}cls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare custom tomato dataset for detect-and-reason")
    parser.add_argument("--classes", type=int, choices=[1, 2], default=2, help="클래스 수 (1=tomato, 2=ripe/unripe)")
    parser.add_argument(
        "--format",
        choices=["yolo", "coco", "both"],
        default="both",
        help="yolo=YOLO 출력만, coco=RF-DETR layout만, both=둘 다",
    )
    parser.add_argument("--force", action="store_true", help="출력 디렉터리를 삭제하고 다시 생성")
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="이미지 연결 방식",
    )
    return parser.parse_args()


def _ensure_source_exists() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Custom tomato dataset not found: {DATASET_ROOT}\n"
            "Create symlink:\n"
            "  ln -s /data/dongyub/custom_tomato_dataset data/custom_tomato_dataset"
        )
    coco_ann = COCO_ROOT / "_annotations.coco.json"
    if not coco_ann.is_file():
        raise FileNotFoundError(f"COCO annotation not found: {coco_ann}")
    for split in SPLITS:
        if not (YOLO_ROOT / "images" / split).is_dir():
            raise FileNotFoundError(f"YOLO split not found: {YOLO_ROOT / 'images' / split}")


def _place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _load_split_stems() -> dict[str, set[str]]:
    return {split: {p.stem for p in (YOLO_ROOT / "images" / split).glob("*.jpg")} for split in SPLITS}


def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox
    x = max(0.0, min(x, width))
    y = max(0.0, min(y, height))
    w = max(0.0, min(w, width - x))
    h = max(0.0, min(h, height - y))
    return [x, y, w, h]


def _split_coco(coco: dict, split_stems: dict[str, set[str]], num_classes: int) -> dict[str, dict]:
    class_names = CLASS_NAMES_BY_N[num_classes]
    images_by_stem = {Path(img["file_name"]).stem: img for img in coco.get("images", [])}
    split_data: dict[str, dict] = {}

    for split in SPLITS:
        stems = split_stems[split]
        split_images = [images_by_stem[stem] for stem in sorted(stems) if stem in images_by_stem]
        old_to_new_img = {int(img["id"]): idx + 1 for idx, img in enumerate(split_images)}
        image_size_by_id = {int(img["id"]): (int(img["width"]), int(img["height"])) for img in split_images}

        split_annotations: list[dict] = []
        ann_id = 1
        for ann in coco.get("annotations", []):
            old_img_id = int(ann["image_id"])
            if old_img_id not in old_to_new_img:
                continue
            img_w, img_h = image_size_by_id[old_img_id]
            bbox = _clip_bbox([float(v) for v in ann.get("bbox", [])[:4]], img_w, img_h)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            # 1cls는 모든 카테고리를 tomato(0)으로 합친다.
            category_id = 0 if num_classes == 1 else int(ann["category_id"])
            split_annotations.append(
                {
                    "id": ann_id,
                    "image_id": old_to_new_img[old_img_id],
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            ann_id += 1

        new_images = [
            {
                "id": idx,
                "file_name": Path(img["file_name"]).name,
                "width": int(img["width"]),
                "height": int(img["height"]),
            }
            for idx, img in enumerate(split_images, start=1)
        ]
        split_data[split] = {
            "info": {"description": f"custom tomato {num_classes}cls - {split}"},
            "images": new_images,
            "annotations": split_annotations,
            "categories": [
                {"id": idx, "name": name, "supercategory": "tomato"}
                for idx, name in enumerate(class_names)
            ],
        }
        print(f"[custom_tomato] coco {split}: {len(new_images)} images, {len(split_annotations)} annotations")
    return split_data


def write_yolo_wrapper(num_classes: int) -> Path:
    """2cls: 외부 YOLO 데이터를 가리키는 data.yaml wrapper 생성 (이미지/라벨 복사 없음)."""
    class_names = CLASS_NAMES_BY_N[num_classes]
    out = yolo_out_dir(num_classes)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / "data.yaml"
    data = {
        "path": str(YOLO_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    print(f"[custom_tomato] YOLO wrapper: {yaml_path}")
    return out


def write_yolo_1cls(split_stems: dict[str, set[str]], force: bool, link_mode: str) -> Path:
    """1cls: 외부 YOLO 이미지를 연결하고 라벨을 모두 class 0(tomato)으로 합쳐 실체 복사본 생성."""
    out = yolo_out_dir(1)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} already exists. Use --force to rebuild.")
        shutil.rmtree(out)

    for split in SPLITS:
        img_dst = out / split / "images"
        lbl_dst = out / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for stem in sorted(split_stems[split]):
            src_img = YOLO_ROOT / "images" / split / f"{stem}.jpg"
            if not src_img.is_file():
                continue
            _place_file(src_img, img_dst / f"{stem}.jpg", link_mode)

            src_lbl = YOLO_ROOT / "labels" / split / f"{stem}.txt"
            lines: list[str] = []
            if src_lbl.is_file():
                for line in src_lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    parts[0] = "0"  # 모든 클래스를 tomato(0)으로
                    lines.append(" ".join(parts))
            (lbl_dst / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = out / "data.yaml"
    data = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "tomato"},
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    print(f"[custom_tomato] YOLO 1cls: {out}")
    return out


def export_rfdetr(split_data: dict[str, dict], num_classes: int, force: bool, link_mode: str) -> Path:
    out = coco_out_dir(num_classes)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} already exists. Use --force to rebuild.")
        shutil.rmtree(out)

    src_images = COCO_ROOT / "images"
    for split in SPLITS:
        split_root = out / RF_SPLIT_NAMES[split]
        images_dir = split_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in split_data[split]["images"]:
            src = src_images / img["file_name"]
            if not src.is_file():
                raise FileNotFoundError(f"Image missing: {src}")
            _place_file(src, images_dir / img["file_name"], link_mode)
        with (split_root / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
            json.dump(split_data[split], handle)

    print(f"[custom_tomato] RF-DETR layout: {out}")
    return out


def main() -> None:
    args = parse_args()
    _ensure_source_exists()

    export_yolo = args.format in {"yolo", "both"}
    export_rfdetr_flag = args.format in {"coco", "both"}
    split_stems = _load_split_stems()

    if export_yolo:
        if args.classes == 1:
            write_yolo_1cls(split_stems, force=args.force, link_mode=args.link_mode)
        else:
            write_yolo_wrapper(args.classes)

    if export_rfdetr_flag:
        with (COCO_ROOT / "_annotations.coco.json").open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        split_data = _split_coco(coco, split_stems, args.classes)
        export_rfdetr(split_data, args.classes, force=args.force, link_mode=args.link_mode)

    print("[custom_tomato] Done.")


if __name__ == "__main__":
    main()
