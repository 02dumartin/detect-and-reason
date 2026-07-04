from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_yaml(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def export_family_predictions(
    *,
    family: str,
    predictions: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if family in {"rf_detr", "dino"}:
        exported = _export_coco(predictions=predictions, output_path=output_dir / "prediction_class_coco.json")
        return {"prediction_class_coco": str(output_dir / "prediction_class_coco.json"), "exported": exported}

    labels_dir = output_dir / "prediction_class_yolo" / "labels"
    exported = _export_yolo(predictions=predictions, labels_dir=labels_dir)
    return {"prediction_class_yolo": str(labels_dir.parent), "exported": exported}


def _export_yolo(*, predictions: list[dict[str, Any]], labels_dir: Path) -> int:
    labels_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[str]] = {}
    exported = 0
    for row in predictions:
        class_id = int(row.get("predicted_class_id", -1))
        if class_id < 0:
            continue
        xc, yc, box_w, box_h = [float(value) for value in row["bbox_yolo"]]
        line = f"{class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f} {float(row.get('det_score', 1.0)):.6f}"
        grouped.setdefault(Path(row["file_name"]).stem, []).append(line)
        exported += 1

    for stem, lines in grouped.items():
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return exported


def _export_coco(*, predictions: list[dict[str, Any]], output_path: Path) -> int:
    rows = []
    for row in predictions:
        class_id = int(row.get("predicted_class_id", -1))
        image_id = row.get("image_id")
        if class_id < 0 or image_id is None:
            continue
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": class_id,
                "bbox": [float(value) for value in row["bbox_xywh"]],
                "score": float(row.get("det_score", 1.0)),
            }
        )
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(rows)

