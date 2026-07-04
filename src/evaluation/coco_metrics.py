from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def _annotation_path_with_info(ann_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    if "info" in data:
        return ann_path, None
    data["info"] = {"description": "COCO annotations"}
    tmp = tempfile.TemporaryDirectory()
    fixed_path = Path(tmp.name) / ann_path.name
    fixed_path.write_text(json.dumps(data), encoding="utf-8")
    return fixed_path, tmp


def coco_map_from_json(
    ann_path: Path,
    pred_path: Path,
) -> Tuple[Optional[float], Optional[float]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_ann_path, tmp_dir = _annotation_path_with_info(ann_path)
    try:
        coco_gt = COCO(str(coco_ann_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0]), float(evaluator.stats[1])
