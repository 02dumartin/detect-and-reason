"""SoM(Set-of-Mark) + mosaic VLM 분류 공용 유틸.

`llm-classify/som_common.py`를 DaR 컨벤션에 맞게 포팅한 것:
- 데이터 로딩은 `som_dataset.py`(DaR의 config/dataset, data/coco/ 재사용)로 대체
- 라벨 정규화는 새로 만들지 않고 `src.vlm.schema.normalize_class_name`(fully-ripe/semi-ripe/unripe
  canonical) 재사용
- 매칭/스코어링(match_det, det_score)은 `src.evaluation.llm_classify_metric` 재사용 (중복 금지)

SoM/mosaic 이미지 생성, staged VQA(dedup/filt/unripe/ripe_semi) 프롬프트, encode_sft,
generate/generate_batch 등 학습·추론에 필요한 나머지는 원본 로직을 그대로 유지한다.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.evaluation.llm_classify_metric import det_score, iou_xyxy, match_det  # noqa: F401 (재export)
from src.vlm.schema import normalize_class_name

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# normalize_class_name()이 어떤 데이터셋의 GT 라벨이든 이 canonical 이름들로 normalize해준다.
# 기본은 3cls(ag_rob_tomato/little/tomatod) 기준이고, 2cls 데이터셋(custom_tomato/rob2pheno,
# ripe/unripe만 있음)에서는 CLI가 시작할 때 set_class_list(("fully-ripe","unripe"))로
# 바꿔서 쓴다 -- "unripe"는 정규화 결과가 같으므로 그대로, 대신 semi-ripe 관련 질문 단계
# (stage_qa의 4번째 task)를 건너뛴다. class_list는 이 모듈 전역에서 참조하는 함수들
# (SomPrompts, pick_exemplars, det_score 호출부 등)이 매번 이 변수를 읽으므로, 데이터셋을
# 바꿀 때마다 다시 import할 필요 없이 set_class_list() 한 번으로 전부 반영된다.
TOMATO_CLASSES = ("fully-ripe", "semi-ripe", "unripe")


def set_class_list(classes: tuple[str, ...]) -> None:
    """이 프로세스가 다룰 ripeness 클래스 목록을 바꾼다 (2cls: ("fully-ripe","unripe")).

    호출한 뒤로는 이 모듈의 모든 `TOMATO_CLASSES` 참조(매칭/스코어링/ICL exemplar 선택/
    `SomPrompts`)가 새 목록을 따른다. 데이터셋당 프로세스 시작 시 한 번만 호출할 것 --
    zeroshot_som.py/sft_som.py/rl_som.py의 main()에서 --dataset을 보고 호출한다.
    """
    global TOMATO_CLASSES
    classes = tuple(classes)
    assert classes in (("fully-ripe", "semi-ripe", "unripe"), ("fully-ripe", "unripe")), (
        f"unsupported class_list: {classes!r} (3cls: fully-ripe/semi-ripe/unripe, "
        "2cls: fully-ripe/unripe만 지원)"
    )
    TOMATO_CLASSES = classes
SIZE = 768
MARKER_STYLE = "box"  # "box"(테두리+모서리 번호, 기존) | "id_only"(테두리 없음, 중앙 번호) | "box_center"(테두리+중앙 번호)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# 오버레이용: 리포 공유 코드가 원래 쓰려던 Noto Sans. 시스템에 설치가 안 돼 있고
# /usr/share/fonts는 쓰기 권한이 없어서(2026-09-11 확인), google/fonts 저장소에서 받아
# 프로젝트 안에 넣어둔 사본을 쓴다.
OVERLAY_FONT_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "fonts" / "NotoSans-Regular.ttf")
FONT = ImageFont.truetype(FONT_PATH, 22)
SMALL_FONT = ImageFont.truetype(FONT_PATH, 10)  # draw_som 박스 번호 뱃지, 16 -> 14 -> 12 -> 10pt
COLORS = [
    (255, 56, 56), (56, 200, 80), (56, 120, 255), (255, 200, 40),
    (200, 80, 255), (40, 220, 220), (255, 120, 40), (180, 255, 60),
]

NOT_TOMATO = "not_tomato"
DUP = "dup"


def set_marker_style(style: str) -> None:
    global MARKER_STYLE
    assert style in ("box", "id_only", "box_center"), f"unknown marker_style: {style!r} (box | id_only | box_center)"
    MARKER_STYLE = style
    print(f"marker_style={style}")


IMAGE_ADJUSTMENT: float | None = None  # brightness factor (예: 1.2=20% 밝게, 0.8=20% 어둡게) | None=미적용


def set_image_adjustment(brightness: float | None) -> None:
    """이미지 로드 시점에 적용할 밝기 보정 계수를 프로세스 전역으로 설정한다.

    데이터셋별로 조명 편향이 달라(tomatod는 전반적으로 어둡게, ag_rob_tomato는 대비가
    강해 밝은 쪽이 과도하게 밝게 찍힘, 2026-09-15 확인) 같은 프롬프트/모델로도 색 판단이
    쏠리는 경향이 있었다. CLAHE는 대비를 과도하게 왜곡해 오히려 색 정보를 망가뜨려서
    (2026-09-15 확인) 단순 밝기 보정(`apply_brightness_contrast`, contrast는 그대로 1.0)만
    쓴다. `open_rgb()`가 로드 직후 적용하므로 SoM/모자이크/exemplar crop 전부 일관되게
    보정된 이미지를 본다 — 학습과 평가가 항상 같은 보정을 거친다.
    """
    global IMAGE_ADJUSTMENT
    IMAGE_ADJUSTMENT = brightness
    print(f"image_adjustment brightness={brightness}")


def norm_label(v: Any) -> str | None:
    """VLM이 뱉은 라벨 문자열을 canonical tag로 정규화한다.

    ripeness는 `src.vlm.schema.normalize_class_name`(fully-ripe/semi-ripe/unripe)을 쓰고,
    tomato 여부/중복 판정은 이 모듈에서 직접 처리한다.
    """
    if not isinstance(v, str):
        v = str(v)
    raw = v.strip().lower().replace(" ", "_")
    if raw in {"dup", "duplicate", "same"}:
        return DUP
    if raw in {"not_tomato", "none", "no_tomato", "other", "small", "tiny", "too_small"}:
        return NOT_TOMATO
    normalized = normalize_class_name(v)
    return normalized if normalized != "none" else NOT_TOMATO


def extract_json(s: str) -> dict:
    s = s.replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(s)
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError(s)


def parse_mark_map(text: str, n: int) -> dict[int, str]:
    obj = extract_json(text)
    if isinstance(obj.get("labels"), dict):
        obj = obj["labels"]
    out: dict[int, str] = {}
    for k, v in obj.items():
        m = re.search(r"\d+", str(k))
        if not m:
            continue
        idx = int(m.group())
        lab = norm_label(v)
        if lab:
            out[idx] = lab
    return {i: out.get(i, "parse_error") for i in range(1, n + 1)}


def open_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) or img
    img = img.convert("RGB")
    if IMAGE_ADJUSTMENT is not None:
        from src.vlm.image_adjustment import apply_brightness_contrast

        img = apply_brightness_contrast(img, brightness=IMAGE_ADJUSTMENT, contrast=1.0)
    return img


def too_small(box: list[int], wh: tuple[int, int], small_rel: float, small_side: float) -> bool:
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    iw, ih = max(1, wh[0]), max(1, wh[1])
    return (w * h) / (iw * ih) < small_rel or min(w, h) / min(iw, ih) < small_side


def label_props(
    item: dict[str, Any],
    boxes: list[list[float]],
    small_rel: float,
    small_side: float,
    thr: float = 0.5,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """proposal 박스(YOLO 1cls 등)를 GT와 IoU 매칭해 (keep/dup/small/other) 라벨을 붙인다."""
    labels: list[str] = []
    used: set[int] = set()
    kept: list[list[int]] = []
    for pb in boxes:
        pb_i = [int(x) for x in pb]
        if too_small(pb_i, item["wh"], small_rel, small_side):
            labels.append(NOT_TOMATO)
            continue
        best_free, biou = -1, thr
        best_any, a_iou = -1, thr
        for i, gb in enumerate(item["boxes"]):
            v = iou_xyxy(pb_i, gb)
            if v > a_iou:
                best_any, a_iou = i, v
            if i not in used and v > biou:
                best_free, biou = i, v
        if best_free >= 0:
            used.add(best_free)
            labels.append(norm_label(item["labels"][best_free]))
            kept.append(pb_i)
        elif best_any >= 0 or any(iou_xyxy(pb_i, kb) >= 0.5 for kb in kept):
            labels.append(DUP)
        else:
            labels.append(NOT_TOMATO)
    out = dict(item)
    out["boxes"] = [list(map(int, b)) for b in boxes]
    out["labels"] = labels
    return shuffle_marks(out, rng or random.Random(_item_seed(item, 3)))


def gt_labeled_item(item: dict[str, Any]) -> dict[str, Any]:
    """ICL exemplar용: GT 박스를 그대로 마크로 써서 "정답 시연"을 만든다.

    `label_props`는 detector proposal의 노이즈(작은 오검출 등)를 걸러내려고
    `too_small` 필터를 적용하는데, ag_rob_tomato처럼 실제 GT 인스턴스 자체가
    화면 대비 작은 경우(온실 전경, 익은 토마토 소수) 이 필터가 진짜 GT까지
    지워버려 exemplar 후보가 하나도 안 남는 문제가 있었다(2026-09-11 확인).
    exemplar는 검출 품질이 아니라 "정답"을 보여주는 용도라 이 필터를 아예 안 쓴다.
    """
    out = dict(item)
    out["boxes"] = [list(map(int, b)) for b in item["boxes"]]
    out["labels"] = [norm_label(l) for l in item["labels"]]
    return shuffle_marks(out, random.Random(_item_seed(item, 3)))


def shuffle_marks(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    idx = list(range(len(item["boxes"])))
    rng.shuffle(idx)
    out = dict(item)
    out["boxes"] = [item["boxes"][i] for i in idx]
    out["labels"] = [item["labels"][i] for i in idx]
    return out


def oversample_by_class(
    train_items: list[dict[str, Any]], raw_by_path: dict[str, dict[str, Any]], max_factor: int = 8,
) -> list[dict[str, Any]]:
    """희귀 클래스를 포함한 이미지를 반복 삽입해서 클래스 불균형을 완화한다.

    sft_som.py/rl_som.py의 학습 루프는 `train_items[step % len(train_items)]`로 원본
    순서를 고정 반복하는 커리큘럼이라, fully-ripe처럼 이미지의 5%에만 등장하는 클래스는
    한 에폭 내내 거의 안 뽑힌다(2026-09-14 확인, ag_rob_tomato: train 348장 중 fully-ripe
    포함 20장뿐이라 SFT 전 스텝에서 fully-ripe F1=0.000). 이미지가 포함한 클래스 중
    최다 클래스 대비 가장 희소한 클래스의 상대 빈도만큼 그 이미지를 반복 삽입해서 한
    에폭 안에서 희귀 클래스를 볼 기회를 늘린다. `max_factor`로 과도한 반복(중복 학습으로
    과적합)을 막고, 마지막에 고정 시드로 섞어서 원본 순서 편향(초반/후반에 특정 클래스
    몰림)도 없앤다. GT 라벨(`raw_by_path`, proposal 매칭 전 원본)을 기준으로 판단한다 --
    proposal 매칭 라벨은 노이즈가 있어 기준으로 쓰기엔 부적절하다.
    """
    from collections import Counter

    class_img_count: Counter[str] = Counter()
    item_classes: list[set[str]] = []
    for it in train_items:
        raw = raw_by_path.get(it["path"])
        classes = {norm_label(lab) for lab in raw["labels"]} if raw else set()
        classes = {c for c in classes if c in TOMATO_CLASSES}
        item_classes.append(classes)
        class_img_count.update(classes)

    if not class_img_count:
        return train_items
    max_count = max(class_img_count.values())

    oversampled: list[dict[str, Any]] = []
    for it, classes in zip(train_items, item_classes):
        factor = 1
        for c in classes:
            factor = max(factor, min(max_factor, round(max_count / class_img_count[c])))
        oversampled.extend([it] * factor)

    random.Random(0).shuffle(oversampled)
    return oversampled


def _item_seed(item: dict[str, Any], salt: int = 0) -> int:
    return (sum(map(ord, str(item.get("path", "")))) * 131 + len(item["boxes"]) * 17 + salt) & 0xFFFFFFFF


def _scale_boxes(boxes: list[list[int]], wh: tuple[int, int], size: int = SIZE) -> list[list[int]]:
    sx, sy = size / wh[0], size / wh[1]
    return [[int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)] for x1, y1, x2, y2 in boxes]


def _letterbox(orig: Image.Image, size: int = SIZE) -> tuple[Image.Image, float, int, int]:
    """정사각형으로 눌러 찌그러뜨리지 않고, 원본 비율을 유지한 채 size×size 캔버스
    (검은 여백 패딩)에 맞춘다. box 좌표를 같은 방식으로 옮기려면 scale/pad_x/pad_y가
    필요해서 같이 반환한다."""
    w, h = orig.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(orig.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _letterbox_scale_boxes(boxes: list[list[int]], scale: float, pad_x: int, pad_y: int) -> list[list[int]]:
    return [
        [int(x1 * scale + pad_x), int(y1 * scale + pad_y), int(x2 * scale + pad_x), int(y2 * scale + pad_y)]
        for x1, y1, x2, y2 in boxes
    ]


def _badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color, font) -> None:
    x, y = xy
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    pad = 4
    draw.rounded_rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], radius=6, fill=color)
    draw.text((x + pad, y + pad - 1), text, fill=(0, 0, 0), font=font)


def draw_som(item: dict[str, Any], size: int = SIZE, orig: Image.Image | None = None, marker_style: str | None = None) -> Image.Image:
    """SoM 이미지: 원본(비율 유지, 레터박스) + 마크.

    marker_style:
    - "box": 테두리(모서리에 번호). rectangle이 열매 표면 일부를 가리는 대신 가림/중복
      판단에 유리하다.
    - "id_only": 테두리 없이 번호 뱃지만(박스 중앙). 색 판단을 안 가리는 대신 마크 경계가
      덜 명확하다.
    - "box_center": 테두리(번호와 같은 색) + 번호는 박스 중앙. 경계는 여전히 보이면서
      번호 뱃지 자체가 과실 가장자리를 가리는 걸 줄인다.
    기본은 MARKER_STYLE 전역(set_marker_style로 변경).
    """
    marker_style = marker_style or MARKER_STYLE
    img, scale, pad_x, pad_y = _letterbox(orig or open_rgb(item["path"]), size)
    boxes = _letterbox_scale_boxes(item["boxes"], scale, pad_x, pad_y)
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = COLORS[i % len(COLORS)]
        if marker_style in ("box", "box_center"):
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        if marker_style == "box":
            cx = max(2, min(x1 + 2, size - 36))
            cy = max(2, min(y1 + 2, size - 24))
        else:
            cx = max(2, min((x1 + x2) // 2 - 12, size - 36))
            cy = max(2, min((y1 + y2) // 2 - 10, size - 24))
        _badge(draw, (cx, cy), str(i + 1), c, SMALL_FONT)
    return img


def make_mosaic(item: dict[str, Any], size: int = SIZE, orig: Image.Image | None = None) -> Image.Image:
    """mosaic 이미지: 크롭 격자, 순서 셔플, 숫자만 매칭.

    box가 하나도 없는 이미지(예: cross-domain proposal이 아무것도 못 찾은 경우)는 크롭할
    대상이 없으므로 빈 캔버스를 그대로 반환한다 -- 이전엔 `max(1, 0)`으로 grid는
    1칸 만들어놓고 실제로는 boxes[0]에 접근해 IndexError가 났었다(2026-09-13).
    """
    orig = orig or open_rgb(item["path"])
    if not item["boxes"]:
        return Image.new("RGB", (size, size), (0, 0, 0))
    n = len(item["boxes"])
    grid = math.ceil(math.sqrt(n))
    cell = size // grid
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    order = list(range(n))
    random.Random(_item_seed(item, 7)).shuffle(order)
    for pos, i in enumerate(order):
        r, c = divmod(pos, grid)
        crop = orig.crop(item["boxes"][i]).resize((cell - 6, cell - 6), Image.BILINEAR)
        d = ImageDraw.Draw(crop)
        col = COLORS[i % len(COLORS)]
        d.rectangle([0, 0, crop.size[0] - 1, crop.size[1] - 1], outline=col, width=3)
        _badge(d, (4, 4), str(i + 1), col, FONT)
        canvas.paste(crop, (c * cell + 3, r * cell + 3))
    return canvas


def pair_images(item: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    orig = open_rgb(item["path"])
    return draw_som(item, orig=orig), make_mosaic(item, orig=orig)


def som_user_content(som: Image.Image, mosaic: Image.Image, text: str) -> list[dict[str, Any]]:
    return [
        {"type": "image", "image": som},
        {"type": "image", "image": mosaic},
        {"type": "text", "text": text},
    ]


def _paint_boxes(orig: Image.Image, boxes_labels: list[tuple[list[int], str]]) -> Image.Image:
    """DaR 표준 overlay 스타일(`src.visualization.yolo_overlay_renderer`)로 박스를 그린다.

    색상/폰트/라벨 배치를 리포 전체 overlay(`render_yolo_overlays.py` 등)와 동일하게 맞춰서
    SoM 결과물만 다르게 보이지 않게 한다.
    """
    from src.visualization.yolo_overlay_renderer import DEFAULT_COLOR_MAP_RGB, DEFAULT_FONT_PATH, _draw_detection, _load_font

    # DEFAULT_FONT_PATH(Noto Sans)가 이 서버엔 실제로 설치돼 있지 않아 _load_font가
    # 조용히 PIL 기본 비트맵 폰트로 fallback하는 문제가 있었다(2026-09-11 확인) — 리포에
    # 이미 있고 실제로 로드되는 DejaVu Sans Bold(FONT_PATH)로 대신 쓴다.
    im = orig.copy()
    draw = ImageDraw.Draw(im)
    font = _load_font(OVERLAY_FONT_PATH, 28)
    for box, cls in boxes_labels:
        _draw_detection(
            draw=draw, bbox_xyxy=box, label=cls, score=1.0,
            color=DEFAULT_COLOR_MAP_RGB.get(cls, (180, 180, 180)),
            font=font, box_thickness=4, label_only=True,
        )
    return im


def _titled_panel(im: Image.Image, title: str) -> Image.Image:
    from src.visualization.yolo_overlay_renderer import _load_font

    top = 34
    canvas = Image.new("RGB", (im.size[0], im.size[1] + top), (18, 20, 22))
    ImageDraw.Draw(canvas).text((8, 5), title, fill=(255, 255, 255), font=_load_font(OVERLAY_FONT_PATH, 18))
    canvas.paste(im, (0, top))
    return canvas


def _gt_vs_pred_row(image_path: str, gt_boxes_labels: list[tuple[list[int], str]], pred_boxes_labels: list[tuple[list[int], str]]) -> Image.Image:
    """[GT | 예측] 두 패널을 나란히 붙인다. 실시간 경로와 predictions.json 기반 경로가 공유한다."""
    orig = open_rgb(image_path)
    left, right = _titled_panel(_paint_boxes(orig, gt_boxes_labels), "GT"), _titled_panel(_paint_boxes(orig, pred_boxes_labels), "prediction")
    gap = 8
    row = Image.new("RGB", (left.size[0] + gap + right.size[0], max(left.size[1], right.size[1])), (18, 20, 22))
    row.paste(left, (0, 0))
    row.paste(right, (left.size[0] + gap, 0))
    return row


def render_gt_vs_pred_overlay(raw_item: dict[str, Any], prop_item: dict[str, Any], pred_map: dict[int, str]) -> Image.Image:
    """[GT | 예측] 오버레이 한 장을 raw_item/prop_item/pred_map(실시간 eval 중간 데이터)에서 만든다."""
    gt = [(box, norm_label(lab)) for box, lab in zip(raw_item["boxes"], raw_item["labels"]) if norm_label(lab) in TOMATO_CLASSES]
    pred = [
        (prop_item["boxes"][j], pred_map.get(j + 1))
        for j in range(len(prop_item["boxes"]))
        if pred_map.get(j + 1) in TOMATO_CLASSES
    ]
    return _gt_vs_pred_row(raw_item["path"], gt, pred)


def dump_predictions_json(
    path: Path,
    raw_items: list[dict[str, Any]],
    prop_items: list[dict[str, Any]],
    pred_maps: list[dict[int, str]],
    raw_texts: list[str],
) -> None:
    """전체 이미지의 GT/proposal/예측/원본 응답을 하나의 json에 남긴다.

    evaluation(`evaluate_predictions_json`)과 오버레이(`render_overlays_from_predictions_json`)가
    둘 다 이 파일 하나에서 재현 가능하게 만들어서, 모델을 다시 안 돌려도 지표·그림을
    다시 만들 수 있게 한다.
    """
    items = []
    for raw, prop, pred, text in zip(raw_items, prop_items, pred_maps, raw_texts):
        items.append(
            {
                "stem": Path(raw["path"]).stem,
                "path": raw["path"],
                "gt": [{"box": box, "label": norm_label(lab)} for box, lab in zip(raw["boxes"], raw["labels"])],
                "proposal_boxes": prop["boxes"],
                "pred_labels": {str(k): v for k, v in pred.items()},
                "raw_response": text,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False))


def _boxes_labels_from_predictions_item(item: dict[str, Any]) -> tuple[list[tuple[list[int], str]], list[tuple[list[int], str]]]:
    gt = [(g["box"], g["label"]) for g in item["gt"] if g["label"] in TOMATO_CLASSES]
    proposal_boxes = item["proposal_boxes"]
    pred = [
        (proposal_boxes[int(idx) - 1], label)
        for idx, label in item["pred_labels"].items()
        if label in TOMATO_CLASSES and 0 < int(idx) <= len(proposal_boxes)
    ]
    return gt, pred


def evaluate_predictions_json(path: Path) -> dict[str, Any]:
    """predictions.json 하나에서 llm-classify 스타일 acc/macro-F1을 다시 계산한다 (모델 불필요)."""
    payload = json.loads(Path(path).read_text())
    pairs: list[tuple[str, str]] = []
    for item in payload["items"]:
        gt, pred_labels_only = _boxes_labels_from_predictions_item(item)
        preds_with_score = [(box, label, 1.0) for box, label in pred_labels_only]
        pairs.extend(match_det(gt, preds_with_score))
    return det_score(pairs, list(TOMATO_CLASSES))


def evaluate_predictions_json_detailed(path: Path, iou_threshold: float = 0.5) -> dict[str, Any]:
    """`evaluate_predictions_json`과 같은 global acc/macro-F1(FP까지 반영하는 llm-classify
    스타일)에 더해, baseline VLM ablation의 `evaluation/` 아티팩트(per_image_metrics.csv,
    confusion_matrix.csv)와 같은 포맷으로 남길 수 있도록 per-image 지표와 confusion matrix까지
    낸다. 여기에 더해 DaR 논문 표(Table I/VIII)의 Loc.Acc/Cls.Acc/Overall Acc — GT 개수를
    분모로 쓰는 recall형 지표라 FP를 못 보는 대신, 그 표의 다른 supervised detector 행과
    바로 비교 가능한 지표 — 도 같이 계산해 둔다(2026-09-13 요청, "loc acc/cls acc가 안 나온다").
    """
    from src.evaluation.yolo_prediction_evaluator import _match_boxes  # noqa: E402 (torch 지연 로드)

    payload = json.loads(Path(path).read_text())
    class_list = list(TOMATO_CLASSES)
    all_pairs: list[tuple[str, str]] = []
    per_image_rows: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, int]] = {}
    total_gt = matched_gt = correct_class_matches = 0

    for item in payload["items"]:
        gt, pred_labels_only = _boxes_labels_from_predictions_item(item)
        preds_with_score = [(box, label, 1.0) for box, label in pred_labels_only]
        pairs = match_det(gt, preds_with_score)
        all_pairs.extend(pairs)

        n_correct = sum(g == p for g, p in pairs)
        per_image_rows.append(
            {
                "stem": item["stem"],
                "n_gt": len(gt),
                "n_pred": len(pred_labels_only),
                "n_pairs": len(pairs),
                "n_correct": n_correct,
                "n_miss": sum(p == "miss" for _, p in pairs),
                "n_fp": sum(g == "none" for g, _ in pairs),
                "acc": n_correct / len(pairs) if pairs else None,
            }
        )
        for gt_cls, pred_cls in pairs:
            confusion.setdefault(gt_cls, {}).setdefault(pred_cls, 0)
            confusion[gt_cls][pred_cls] += 1

        # Loc/Cls/Overall Acc: 클래스 무시하고 IoU만으로 먼저 매칭한 뒤 클래스를 본다
        # (match_det/llm-classify는 클래스까지 같아야 매칭이 성립해서 별도로 다시 잰다).
        gt_boxes_only = [box for box, _ in gt]
        pred_boxes_only = [box for box, _ in pred_labels_only]
        matches, _, _ = _match_boxes(gt_boxes_only, pred_boxes_only, iou_threshold)
        total_gt += len(gt)
        matched_gt += len(matches)
        correct_class_matches += sum(gt[gi][1] == pred_labels_only[pi][1] for gi, pi, _ in matches)

    score = det_score(all_pairs, class_list)
    score["per_image"] = per_image_rows
    score["confusion"] = confusion
    score["num_images"] = len(payload["items"])
    score["loc_acc"] = matched_gt / total_gt if total_gt else 0.0
    score["cls_acc"] = correct_class_matches / matched_gt if matched_gt else 0.0
    score["overall_acc"] = correct_class_matches / total_gt if total_gt else 0.0
    return score


def render_overlays_from_predictions_json(path: Path, output_dir: Path, max_images: int | None = None) -> int:
    """predictions.json에서 [GT | 예측] 오버레이를 렌더링한다 (모델 불필요). 저장한 장수를 반환."""
    payload = json.loads(Path(path).read_text())
    items = payload["items"][:max_images] if max_images is not None else payload["items"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(items):
        gt, pred = _boxes_labels_from_predictions_item(item)
        _gt_vs_pred_row(item["path"], gt, pred).save(output_dir / f"{idx:02d}_{item['stem']}.jpg", quality=90)
    return len(items)


def save_qwen_input_dump(
    folder: Path,
    prompts: "SomPrompts",
    prop_item: dict[str, Any],
    icl: list[dict[str, Any]],
    raw_response: str,
) -> None:
    """VLM에 실제로 들어간 이미지(SoM/mosaic)와 프롬프트, 원본 응답을 그대로 저장한다.

    llm-classify의 `dump_qwen_inputs.py`가 하던 것과 같은 목적 — 학습/추론 결과가 아니라
    "정말로 이게 모델이 본 전부였다"를 검증용으로 남긴다. GT는 여기 어디에도 없다.
    """
    folder.mkdir(parents=True, exist_ok=True)
    orig, _, _, _ = _letterbox(open_rgb(prop_item["path"]), SIZE)
    som, mosaic = pair_images(prop_item)
    orig.save(folder / "input.jpg", quality=90)
    som.save(folder / "som.jpg", quality=90)
    mosaic.save(folder / "mosaic.jpg", quality=90)

    panels = [_titled_panel(orig, "input"), _titled_panel(som, "SoM"), _titled_panel(mosaic, "mosaic")]
    strip = Image.new("RGB", (sum(p.size[0] for p in panels), max(p.size[1] for p in panels)), (18, 20, 22))
    x = 0
    for panel in panels:
        strip.paste(panel, (x, 0))
        x += panel.size[0]
    strip.save(folder / "strip.jpg", quality=90)

    prompt_text = prompts.with_icl(icl, prop_item)[-1]["content"][-1]["text"]
    (folder / "prompt.txt").write_text(prompt_text)
    (folder / "raw_response.txt").write_text(raw_response)
    try:
        parsed = parse_mark_map(raw_response, len(prop_item["boxes"]))
    except Exception:
        parsed = {}
    (folder / "parsed.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False))


class SomPrompts:
    """이미지 크기 통계로 프롬프트를 만든다 (`apply_size_stats` 대응). 데이터셋별로 인스턴스 하나씩 쓴다."""

    ICL_HINT = (
        "Previous turns are labeled in-context examples. Follow that format. "
        "This is a NEW image; do not copy those labels."
    )

    def __init__(self, train_items: list[dict[str, Any]]):
        rel, side, px = [], [], []
        for it in train_items:
            iw, ih = max(1, it["wh"][0]), max(1, it["wh"][1])
            for b in it["boxes"]:
                w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
                rel.append(w * h / (iw * ih))
                side.append(min(w, h) / min(iw, ih))
                px.append(min(w, h))
        self.small_rel = self._pct(rel, 5)
        self.small_side = self._pct(side, 5)
        self.base_prompt = self._make_prompt(
            len(rel), self.small_rel, self._pct(rel, 50), self._pct(rel, 95),
            self.small_side, self._pct(side, 50), self._pct(px, 5),
        )
        # Qwen3.5 같은 thinking 모델의 native <think> 추론은 토큰 예산을 끝없이 잡아먹어서
        # (마크 하나하나 자연어로 분석하다 max_new_tokens를 다 씀) 최종 JSON을 못 내는 경우가
        # 있었다. DaR의 다른 VLM 파이프라인(src/vlm/prompting.py)처럼, native thinking은 끄고
        # 대신 JSON 스키마 안에 "reasoning" 필드를 두어 몇 단어로 분량을 못박는 방식으로 대체한다.
        self.full_prompt = (
            self.base_prompt + "\n\n" + self.ICL_HINT
            + "\nClassify every mark. Do not reason mark-by-mark in free text — "
            + "put ONE short overall reasoning sentence (Step1: overall color/pattern note, "
            + "Step2: classify), 40 words or fewer, inside the JSON itself. JSON only, no text before/after:\n"
            + '{"reasoning":"Step1: ... Step2: ...","labels":{"1":"ripe","2":"not_tomato"}}'
        )

    @staticmethod
    def _pct(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

    @staticmethod
    def _make_prompt(n, rel5, rel50, rel95, side5, side50, px5) -> str:
        two_class = len(TOMATO_CLASSES) == 2
        ripeness_bullets = (
            "- ripe: mostly red\n- unripe: green, whitish, orange, yellow, or clearly mixed red and green"
            if two_class
            else "- ripe: mostly red\n- semi_ripe: orange, yellow, or clearly mixed red and green\n- unripe: green or whitish"
        )
        return f"""You are a senior greenhouse tomato-ripeness annotator.

For every mark, use BOTH images. Switch between them:
- Image 1 (SoM): full frame with numbered boxes. Use this for cluster context — a tomato behind another tomato still counts.
- Image 2 (mosaic): zoomed crop of that box. Tile order is shuffled; empty cells are black. Match by the number badge, never by grid position.
Mosaic = skin color. SoM = is there a fruit in that box, including fruit hidden behind another.

Ripeness (fruit skin, ignore calyx/leaf):
{ripeness_bullets}
- not_tomato: leaf, stem, flower, twine, or empty background — no fruit at all

Do not let brightness, glossy specular highlight, or shadow confuse the ripeness class.
A green fruit with a bright white highlight is still unripe, not ripe. A red fruit sitting
in shadow is still ripe, not unripe. Judge by hue, not by how bright or dark the mark looks.

"Which ripeness class" and "is there a fruit at all" are two SEPARATE questions — do not
let uncertainty about the first leak into the second. Being unsure of the exact hue, or
only seeing a sliver of skin, a blurry edge, a fruit mostly hidden behind another, or a
patch lit oddly, is ALWAYS a ripeness question: pick your best-guess ripeness class, never
not_tomato. not_tomato is reserved ONLY for marks where you are confident no fruit skin is
present at all in that box — pure leaf, stem, flower, twine, soil, or empty background.
If you can see even a small patch of fruit skin, however partial, occluded, or oddly lit,
you must output a ripeness class, not not_tomato.
Labeled GT size (train n={n}): box-area/image p5={100 * rel5:.3f}% p50={100 * rel50:.2f}% p95={100 * rel95:.1f}%; min-side/min(image) p5={100 * side5:.1f}% p50={100 * side50:.1f}% (~{px5:.0f}px at p5). Annotators deliberately skipped fruit below ~p5. Mosaic zoom makes tiny crops look large — use SoM size and the area list. If a mark is below that p5 range, not_tomato even if it looks like fruit.
If several numbered boxes sit on the SAME fruit in the SoM, keep one tighter box and treat extras as duplicates — that is not two tomatoes.

Do not list marks one by one. Classify this image's numbers."""

    def prompt_ex(self, k: int = 1) -> str:
        return (
            self.base_prompt
            + f"\n\nThis is worked EXAMPLE {k} with gold labels. JSON only:\n"
            + '{"1":"ripe","2":"not_tomato"}'
        )

    def size_note(self, item: dict[str, Any]) -> str:
        iw, ih = max(1, item["wh"][0]), max(1, item["wh"][1])
        xs = []
        for i, b in enumerate(item["boxes"]):
            w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
            xs.append(f"{i + 1}:{100 * w * h / (iw * ih):.2f}%")
        return "This image mark area/image: " + " ".join(xs)

    def stage_qa(self, item: dict[str, Any], kind: str, example: int = 0) -> tuple[str, dict[str, str]]:
        """4단계(3cls: dedup/filt/unripe/ripe_semi) 또는 3단계(2cls: dedup/filt/ripe_unripe)
        staged VQA. 2cls는 semi-ripe 구분이 없어서 unripe-vs-later + ripe-vs-semi 두 단계를
        "tomato 마크 중 ripe vs unripe" 한 단계로 합친다 (`TOMATO_CLASSES` 길이로 분기,
        `set_class_list()`가 프로세스 시작 시 이미 정해둔 값을 읽는다)."""
        if kind == "full":
            return self._full_stage_qa(item)
        two_class = len(TOMATO_CLASSES) == 2
        total = 3 if two_class else 4
        labs = item["labels"]
        hint = f"This is worked EXAMPLE {example} with gold labels." if example else self.ICL_HINT
        if kind == "dedup":
            q = (
                self.base_prompt + "\n\n" + hint
                + f"\nTask 1/{total}: FULL SoM only. If several boxes sit on the SAME fruit, keep the tighter one, extras dup. "
                + "Different fruits that touch are both keep.\n"
                + 'JSON only: {"1":"keep","2":"dup"}'
            )
            a = {str(i + 1): (DUP if lab == DUP else "keep") for i, lab in enumerate(labs)}
            return q, a
        if kind == "filt":
            q = (
                self.base_prompt + "\n\n" + hint
                + f"\nTask 2/{total}: tomato vs not_tomato. Use SoM+mosaic. Conservative not_tomato if no fruit. "
                + "Also not_tomato if the fruit is tiny vs others in the SoM.\n"
                + 'JSON only: {"1":"tomato","2":"not_tomato"}'
            )
            a = {
                str(i + 1): (NOT_TOMATO if lab == NOT_TOMATO else "tomato")
                for i, lab in enumerate(labs) if lab != DUP
            }
            return q, a or {"none": "tomato"}
        if kind == "ripe_unripe":  # 2cls 전용 최종 단계
            q = (
                self.base_prompt + "\n\n" + hint
                + f"\nTask 3/{total}: tomato marks only. ripe vs unripe.\n"
                + 'JSON only: {"1":"ripe","2":"unripe"}'
            )
            a = {
                str(i + 1): ("ripe" if lab == "fully-ripe" else "unripe")
                for i, lab in enumerate(labs) if lab in TOMATO_CLASSES
            }
            return q, a or {"none": "unripe"}
        if kind == "unripe":
            q = (
                self.base_prompt + "\n\n" + hint
                + f"\nTask 3/{total}: tomato marks only. unripe vs later. Keep fruit behind another tomato.\n"
                + 'JSON only: {"1":"unripe","3":"later"}'
            )
            a = {
                str(i + 1): ("unripe" if lab == "unripe" else "later")
                for i, lab in enumerate(labs) if lab in TOMATO_CLASSES
            }
            return q, a or {"none": "later"}
        q = (
            self.base_prompt + "\n\n" + hint
            + f"\nTask 4/{total}: leftover fruit ripe vs semi_ripe. Ignore unripe, not_tomato, dup.\n"
            + 'JSON only: {"1":"ripe","2":"semi_ripe"}'
        )
        a = {
            str(i + 1): lab for i, lab in enumerate(labs)
            if lab in ("fully-ripe", "semi-ripe")
        }
        return q, a or {"none": "fully-ripe"}

    def _full_stage_qa(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """평가 때 실제로 채점하는 것과 완전히 같은 포맷(full_prompt + reasoning-wrapped
        JSON)을 그대로 학습에 쓰는 스테이지.

        기존 4단계(dedup/filt/unripe/ripe_semi) 커리큘럼은 채점 없는 하위 태스크만 연습해서
        SFT/RL이 실제 평가 스키마(`{"reasoning":...,"labels":{...}}`)를 한 번도 훈련 중에
        보지 못했다(2026-09-15 확인 -- 논문 리뷰 중 지적됨). reasoning 문장 자체는 채점 대상이
        아니므로(evaluate_predictions_json은 labels만 봄) 형식만 맞춘 고정 스캐폴드를 쓴다.
        """
        return self.full_prompt, {
            "reasoning": "Step1: observed color/pattern evidence per mark. Step2: classified accordingly.",
            "labels": self.labels_json(item),
        }

    @staticmethod
    def stage_names() -> tuple[str, ...]:
        """이 클래스 수(2/3cls)에서 vqa_pair가 고를 수 있는 stage 이름들.

        RL(`rl_som.py`)이 k개 rollout을 뽑기 전에 stage를 미리 하나 골라 고정할 때 쓴다 —
        안 그러면 매 rollout마다 다시 무작위로 골라서, 서로 다른 난이도의 질문(예: 쉬운
        dedup vs 어려운 ripe_unripe)이 같은 GRPO advantage 그룹에 섞이는 문제가 있었다
        (2026-09-14 확인). `full`은 평가와 동일한 포맷(위 `_full_stage_qa`)을 훈련하는
        스테이지다(2026-09-15 추가).
        """
        return ("dedup", "filt", "ripe_unripe", "full") if len(TOMATO_CLASSES) == 2 else ("dedup", "filt", "unripe", "ripe_semi", "full")

    @staticmethod
    def stage_weights() -> tuple[float, ...]:
        """`stage_names()`와 같은 순서의 샘플링 가중치.

        기존엔 균등 랜덤이라, 오버샘플링(`oversample_by_class`)으로 희귀 클래스 이미지
        노출을 늘려도 정작 그 클래스를 직접 채점하는 ripeness 스테이지(`unripe`/`ripe_semi`/
        `ripe_unripe`)가 걸릴 확률은 그대로라 효과가 스테이지 단계에서 다시 희석됐다
        (2026-09-15 확인, ag_rob_tomato: fully-ripe 6개 중 4개가 semi-ripe로 오분류).
        dedup/filt는 보조 태스크라 가중치를 낮추고, 실제 macro-F1을 결정하는
        ripeness 스테이지와 평가 포맷을 그대로 연습하는 `full`에 가중치를 더 준다.
        """
        return (1.0, 1.0, 3.0, 2.0) if len(TOMATO_CLASSES) == 2 else (1.0, 1.0, 2.0, 3.0, 2.0)

    def vqa_pair(self, item: dict[str, Any], rng: random.Random, kind: str | None = None):
        kind = kind or rng.choices(self.stage_names(), weights=self.stage_weights(), k=1)[0]
        return kind, self.stage_qa(item, kind)

    def icl_stage(self, exemplars: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
        msgs = []
        for k, ex in enumerate(exemplars, 1):
            som, mosaic = pair_images(ex)
            q, a = self.stage_qa(ex, kind, example=k)
            msgs.append({"role": "user", "content": som_user_content(som, mosaic, q)})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(a, ensure_ascii=False)}]})
        return msgs

    def with_icl(self, icl: list[dict[str, Any]], item: dict[str, Any], text: str | None = None, answer=None):
        som, mosaic = pair_images(item)
        body = (text or self.full_prompt) + "\n" + self.size_note(item)
        msgs = list(icl) + [{"role": "user", "content": som_user_content(som, mosaic, body)}]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer, ensure_ascii=False)}]})
        return msgs

    def labels_json(self, item: dict[str, Any]) -> dict[str, str]:
        pub = {NOT_TOMATO: NOT_TOMATO, DUP: DUP}
        return {str(i + 1): pub.get(lab, lab) for i, lab in enumerate(item["labels"])}

    def exemplar_messages(self, ex: dict[str, Any], k: int = 1) -> list[dict[str, Any]]:
        som, mosaic = pair_images(ex)
        ans = self.labels_json(ex)
        return [
            {"role": "user", "content": som_user_content(som, mosaic, self.prompt_ex(k) + "\n" + self.size_note(ex))},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(ans, ensure_ascii=False)}]},
        ]

    def icl_messages(self, exemplars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs = []
        for k, ex in enumerate(exemplars, 1):
            msgs.extend(self.exemplar_messages(ex, k))
        return msgs

    REVIEW_PROMPT = (
        "Re-examine your own answer above. Look again at both images for each numbered mark, "
        "especially any you called not_tomato — if it actually shows even a sliver of fruit "
        "skin, correct it to a ripeness class. Also double check you judged hue, not "
        "brightness/glare/shadow. If everything was already correct, repeat the same JSON "
        "unchanged. Output ONLY the corrected JSON, same schema as before."
    )

    def review_messages(self, base_messages: list[dict[str, Any]], first_pass_text: str) -> list[dict[str, Any]]:
        """1차 응답을 스스로 재검토하게 하는 2차 turn을 붙인다 (자기검증 pass).

        `base_messages`는 그 아이템의 1차 질의에 실제로 쓰인 메시지 리스트
        (`with_icl(icl, item)`의 결과) 그대로 넘겨야 한다 — 같은 이미지를 다시 안 그리고
        재사용한다.
        """
        return list(base_messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": first_pass_text}]},
            {"role": "user", "content": [{"type": "text", "text": self.REVIEW_PROMPT}]},
        ]

    @property
    def RECHECK_NOT_TOMATO_PROMPT(self) -> str:
        ripeness_options = "ripe/unripe" if len(TOMATO_CLASSES) == 2 else "ripe/semi_ripe/unripe"
        return (
            "These numbered marks were judged not_tomato in a first pass. Look again, carefully, "
            "at both images for each one. If you can see even a small patch of fruit skin — "
            f"partial, occluded, blurry, or oddly lit — give its ripeness class ({ripeness_options}"
            ") instead. Only keep not_tomato if you are confident there is truly no fruit "
            "in that box. JSON only, same schema, one label per mark:\n"
            '{"1":"ripe","2":"not_tomato"}'
        )

    def recheck_messages(self, subset: dict[str, Any]) -> list[dict[str, Any]]:
        """`not_tomato` 마크만 골라 다시 그린 subset item에 대한 재확인 질의 1턴."""
        som, mosaic = pair_images(subset)
        return [{"role": "user", "content": som_user_content(som, mosaic, self.RECHECK_NOT_TOMATO_PROMPT)}]


def pick_exemplars(
    items: list[dict[str, Any]],
    label_fn,
    n: int = 2,
    min_tom: int = 1,
) -> list[dict[str, Any]]:
    """ripe+semi+unripe를 exemplar n장의 "합집합"으로 커버하도록 고른다.

    한 장에 3클래스가 다 있으면 이상적이지만, 클래스 불균형이 심한 데이터셋
    (예: ag_rob_tomato는 fully-ripe/semi-ripe/unripe가 같은 학습 이미지 안에 동시에
    등장하는 경우가 0장이었다 — 2026-09-11 확인)에서는 그런 이미지 자체가 존재하지
    않을 수 있다. exemplar의 목적은 "라벨 어휘를 시연"하는 것이지 각 장이 완전한
    데이터셋 축소판일 필요는 없으므로, 새로운 클래스를 가장 많이 추가하는 이미지를
    그리디하게 골라 n장까지의 합집합이 3클래스를 다 덮게 한다.
    """
    need = set(TOMATO_CLASSES)
    labeled = [label_fn(it) for it in items]

    def tom_labels(ex: dict[str, Any]) -> set[str]:
        return set(norm_label(l) for l in ex["labels"]) & need

    def n_tom(ex: dict[str, Any]) -> int:
        return sum(norm_label(l) in need for l in ex["labels"])

    for thr in range(min_tom, 0, -1):
        pool = [ex for ex in labeled if n_tom(ex) >= thr]
        pool.sort(key=n_tom, reverse=True)
        chosen: list[dict[str, Any]] = []
        covered: set[str] = set()
        remaining = list(pool)
        while remaining and len(chosen) < n:
            best = max(remaining, key=lambda ex: len(tom_labels(ex) - covered))
            if not (tom_labels(best) - covered) and covered:
                break  # 더 이상 새 클래스를 못 채우면 그만
            chosen.append(best)
            covered |= tom_labels(best)
            remaining.remove(best)
        if covered >= need:
            # 남은 slot은 다양성을 위해 다음으로 풍부한 후보로 채운다.
            for ex in pool:
                if len(chosen) == n:
                    break
                if ex not in chosen:
                    chosen.append(ex)
            return chosen[:n]
    raise ValueError(f"need {n} ICL images covering {TOMATO_CLASSES} (found union coverage {covered})")


def pick_class_exemplar_crops(
    train_items: list[dict[str, Any]], class_list: tuple[str, ...] | None = None, seed: int = 0
) -> list[tuple[Image.Image, str]]:
    """클래스별 대표 crop 하나씩 골라 (crop, label) 리스트로 반환한다.

    `pick_exemplars`(전체 SoM+mosaic worked example, ICL용)와는 다른 성격의 참조자료다 —
    여기는 마크 전체가 아니라 "이 색이 unripe/semi_ripe/ripe다"를 보여주는 라벨 하나짜리
    작은 크롭만 필요해서, GT 박스를 그대로 잘라서 쓴다.
    """
    class_list = class_list or TOMATO_CLASSES
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for it in train_items:
        for box, lab in zip(it["boxes"], it["labels"]):
            norm = norm_label(lab)
            if norm in class_list:
                by_class[norm].append((it["path"], box))

    out: list[tuple[Image.Image, str]] = []
    for cls in class_list:
        candidates = by_class.get(cls, [])
        if not candidates:
            continue
        path, box = rng.choice(candidates)
        out.append((open_rgb(path).crop(box), cls))
    return out


def class_exemplar_content(crops_with_labels: list[tuple[Image.Image, str]]) -> list[dict[str, Any]]:
    """클래스별 참조 crop을 이미지+캡션 텍스트로 번갈아 넣는 content 블록을 만든다."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Reference examples for color calibration only (not part of the query marks):"}
    ]
    for crop, label in crops_with_labels:
        content.append({"type": "image", "image": crop})
        content.append({"type": "text", "text": f"-> {label}"})
    return content


def exemplar_preamble_messages(crops_with_labels: list[tuple[Image.Image, str]]) -> list[dict[str, Any]]:
    """`icl`(with_icl이 받는 prior turn 리스트) 앞에 붙일 수 있는 참조 crop 전용 turn.

    `with_icl(icl, item, ...)`는 `icl` 리스트를 그대로 앞에 붙이기만 하므로, ICL worked
    example과 독립적으로(또는 함께) 이 turn 하나만 리스트에 넣어도 그대로 동작한다.
    """
    if not crops_with_labels:
        return []
    return [{"role": "user", "content": class_exemplar_content(crops_with_labels)}]


def subset_item(item: dict[str, Any], indices_1based: list[int]) -> dict[str, Any]:
    """1-indexed 마크 번호 목록만 골라 boxes를 재구성한다(새 item에서는 다시 1..k로 번호가 매겨짐).

    `not_tomato` 재확인처럼 일부 마크만 다시 그려서 물어볼 때 쓴다 — `wh`/`path`는 원본을
    그대로 쓰므로 `draw_som`/`make_mosaic`가 별도 처리 없이 그대로 동작한다.
    """
    out = dict(item)
    out["boxes"] = [item["boxes"][i - 1] for i in indices_1based]
    return out


def json_acc(gold: dict[str, Any], text: str) -> float:
    """VQA 스테이지 정답과 모델 응답을 비교해 마크별 정확도를 낸다 (RL reward).

    최종 평가(`parse_mark_map`/`evaluate_predictions_json`)와 기준을 맞추려고 두 가지를
    보정한다(2026-09-15 확인). (1) `full` 스테이지의 정답/예측은 `{"labels":{...}}`로
    감싸져 있는데, 그대로 두면 최상위 키에서 "1","2",...를 못 찾아 보상이 항상 0이 된다 —
    `parse_mark_map`과 동일하게 `"labels"`를 언래핑한다. (2) ripeness 클래스 값은
    `norm_label()`로 비교해서 "ripe"/"fully_ripe"/"fully-ripe" 같은 표기 차이를 최종 평가와
    동일하게 관대히 봐준다 — 다만 dedup/filt 스테이지의 keep/dup/tomato/not_tomato/later
    같은 non-ripeness 토큰까지 `norm_label()`에 넣으면 전부 "no class" 판정으로 뭉개져
    서로 다른 값이 우연히 같아져 버리므로, 그런 토큰은 기존처럼 단순 문자열 비교만 한다.
    """
    try:
        pred = extract_json(text)
    except Exception:
        return 0.0
    if isinstance(pred.get("labels"), dict):
        pred = pred["labels"]
    if isinstance(gold.get("labels"), dict):
        gold = gold["labels"]
    if not gold:
        return 0.0

    non_ripeness_tokens = {"keep", "dup", "duplicate", "same", "tomato", "not_tomato", "no_tomato", "later", "none"}

    def nrm(x):
        s = str(x).lower().replace("-", "_").strip()
        if s in non_ripeness_tokens:
            return s
        return norm_label(x) or s

    ok = sum(1 for k, v in gold.items() if nrm(pred.get(k, pred.get(str(k), ""))) == nrm(v))
    return ok / len(gold)


# --- VLM I/O (transformers 직접 호출, ms-swift 미사용 — raw loop 포팅) ---------------------


def load_vlm(model_id: str = MODEL_ID, device: str = "cuda", merge_lora: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # 아키텍처를 하드코딩(Qwen3VLForConditionalGeneration)하지 않고 Auto 클래스를 쓴다 --
    # Qwen3-VL(4B, sft/RL)과 Qwen3.5(27B, zs 계열과 모델 통일용, transformers>=5.17 필요)가
    # 서로 다른 model_type이라 이 Auto 클래스 하나로 둘 다 커버해야 model_id만 바꿔서
    # 재사용할 수 있다 (2026-09-13, EXPERIMENT_PLAN.md "Qwen3.5-27B로 SFT/RL 통일" 참고).
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device},
    )
    proc = AutoProcessor.from_pretrained(model_id)
    if merge_lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, merge_lora)
        model = model.merge_and_unload()
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return model, proc


def _apply(proc, msgs, gen: bool = True) -> str:
    kw = dict(tokenize=False, add_generation_prompt=gen)
    try:
        return proc.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return proc.apply_chat_template(msgs, **kw)


def generate(model, proc, messages, max_new_tokens: int = 192, do_sample: bool = False, temperature: float = 1.0):
    import torch
    from qwen_vl_utils import process_vision_info

    text = _apply(proc, messages, True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=True)
    if do_sample:
        kw.update(temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return proc.decode(gen_ids, skip_special_tokens=True).strip(), inputs, out


def generate_batch(model, proc, batch, max_new_tokens: int = 192, do_sample: bool = False, temperature: float = 1.0):
    import torch
    from qwen_vl_utils import process_vision_info

    proc.tokenizer.padding_side = "left"
    texts, imgs = [], []
    for msgs in batch:
        texts.append(_apply(proc, msgs, True))
        im, _ = process_vision_info(msgs)
        imgs.extend(im or [])
    inputs = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=True)
    if do_sample:
        kw.update(temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    plen = inputs["input_ids"].shape[1]
    return [proc.decode(row[plen:], skip_special_tokens=True).strip() for row in out]


def auto_bs(has_icl: bool) -> int:
    import torch

    free = torch.cuda.mem_get_info()[0] / 1024**3
    if has_icl:
        return 2 if free >= 22 else 1
    return 6 if free >= 30 else 4 if free >= 18 else 2


def hf_generate_fn(model, proc):
    """HF transformers 모델용 generate_fn 어댑터 (`eval_som_vlm`이 요구하는 시그니처)."""

    def _fn(batch: list[list[dict[str, Any]]], max_new_tokens: int) -> list[str]:
        return generate_batch(model, proc, batch, max_new_tokens=max_new_tokens)

    return _fn


def vllm_generate_fn(backend):
    """`src.vlm.backends.qwen_vllm.QwenVllmBackend`용 generate_fn 어댑터.

    vLLM은 backend 자체 generation_cfg(max_tokens 등)로 sampling params를 이미
    고정해서 만들기 때문에 max_new_tokens 인자는 무시한다 — thinking 모델(Qwen3.5)은
    짧은 태스크에도 토큰을 많이 쓸 수 있어서, 호출 전에 넉넉한 max_new_tokens로
    backend를 만들어 두는 쪽을 권장한다.
    """

    def _fn(batch: list[list[dict[str, Any]]], max_new_tokens: int) -> list[str]:  # noqa: ARG001
        return backend.generate_batch(batch)

    return _fn


def eval_som_vlm(
    generate_fn,
    prompts: "SomPrompts",
    raw_items: list[dict[str, Any]],
    proposals: dict[str, list[list[int]]],
    icl: list[dict[str, Any]],
    bs: int = 4,
    desc: str = "eval",
    predictions_json_path: Path | None = None,
    qwen_inputs_dir: Path | None = None,
    qwen_inputs_max: int | None = None,
    second_pass_review: bool = False,
    recheck_not_tomato: bool = False,
    max_new_tokens_cap: int = 768,
) -> dict[str, Any]:
    """proposal(1cls) 박스에 대해 SoM+mosaic 1회 호출로 전체 분류하고 det_score로 채점한다.

    `generate_fn(batch_of_message_lists, max_new_tokens) -> list[str]` — HF(`hf_generate_fn`)
    든 vLLM(`vllm_generate_fn`)이든 이 시그니처만 맞으면 그대로 쓸 수 있다.

    `predictions_json_path`가 주어지면 전체 이미지의 GT/proposal/예측/원본응답을 json으로
    남긴다 — 이 파일 하나로 나중에 모델 없이 `evaluate_predictions_json`/
    `render_overlays_from_predictions_json`을 다시 돌릴 수 있다.
    `qwen_inputs_dir`가 주어지면 실제 VLM에 들어간 SoM/mosaic 이미지 + 프롬프트 + 원본
    응답을 저장한다 (GT는 여기 안 들어간다 — 모델 입력 그 자체만 남김). `qwen_inputs_max`가
    None(기본)이면 전체 저장, 정수면 앞에서 그만큼만 저장한다. (참고: 이 dump는 1차 응답
    기준이라, 아래 2차 pass들로 라벨이 바뀐 최종 결과와는 raw_response.txt가 다를 수 있다 —
    predictions.json/pred_maps 쪽이 항상 최종 결과다.)

    `second_pass_review=True`면 1차 답변 전체를 모델이 스스로 재검토하는 2차 VLM 호출을
    이미지당 1번 더 추가한다(자기검증). `recheck_not_tomato=True`면 1차에서 `not_tomato`로
    나온 마크만 모아 재확인하는 VLM 호출을 추가한다(애매한 것만 다시 물어서 비용이 더 적음).
    둘 다 켜면 순서대로(review 먼저, 그 다음 recheck) 적용된다.

    `max_new_tokens_cap`: hf_generate_fn 경로(SFT/RL 평가)가 이미지당 그대로 max_new_tokens로
    쓰는 고정 예산. 원래는 `min(512, 32 + 8*marks)`처럼 마크 수에 비례해서 깎았는데, 그
    계수(8 token/mark)가 라벨 한 개("14":"semi_ripe", 약 9~10 token) 값도 못 채울 만큼
    작아서 캡을 512→768로 올린 뒤에도 여전히 대부분의 이미지가 예산에 못 미쳐(캡이 걸리기도
    전에) 응답이 중간에 잘렸다(2026-09-14 확인 — bs=1로 재현해도 동일, tokenizer로 실제
    생성 토큰 수==예산 일치 확인). vllm_generate_fn 경로(zero-shot)는 애초에 이 인자를
    무시하고 backend config의 고정 max_new_tokens(768)를 이미지당 그대로 쓰기 때문에 이
    문제가 없었다 — 두 경로를 동일하게 만들기 위해 마크 수 스케일링을 버리고 이미지당
    고정 `max_new_tokens_cap`을 그대로 쓴다.
    """
    from tqdm import tqdm

    prop_items = [label_props(it, proposals.get(it["path"], []), prompts.small_rel, prompts.small_side) for it in raw_items]
    pred_maps = []
    raw_texts = []
    base_msgs_by_item = []
    for i in tqdm(range(0, len(prop_items), bs), desc=desc):
        chunk = prop_items[i : i + bs]
        chunk_msgs = [prompts.with_icl(icl, it) for it in chunk]
        base_msgs_by_item.extend(chunk_msgs)
        outs = generate_fn(chunk_msgs, max_new_tokens_cap)
        for it, text in zip(chunk, outs):
            raw_texts.append(text)
            try:
                pred_maps.append(parse_mark_map(text, len(it["boxes"])))
            except Exception:
                pred_maps.append({})

    if second_pass_review:
        for i in tqdm(range(0, len(prop_items), bs), desc=f"{desc}-review"):
            idxs = list(range(i, min(i + bs, len(prop_items))))
            review_msgs = [prompts.review_messages(base_msgs_by_item[j], raw_texts[j]) for j in idxs]
            outs = generate_fn(review_msgs, max_new_tokens_cap)
            for j, text in zip(idxs, outs):
                try:
                    new_map = parse_mark_map(text, len(prop_items[j]["boxes"]))
                except Exception:
                    continue
                if new_map:
                    pred_maps[j] = new_map
                    raw_texts[j] = text

    if recheck_not_tomato:
        recheck_jobs = []  # (item_idx, [원래 1-idx, ...], subset item)
        for j, (it, pred) in enumerate(zip(prop_items, pred_maps)):
            flagged = [k for k in range(1, len(it["boxes"]) + 1) if pred.get(k) == NOT_TOMATO]
            if flagged:
                recheck_jobs.append((j, flagged, subset_item(it, flagged)))
        for i in tqdm(range(0, len(recheck_jobs), bs), desc=f"{desc}-recheck"):
            chunk = recheck_jobs[i : i + bs]
            msgs = [prompts.recheck_messages(sub) for _, _, sub in chunk]
            outs = generate_fn(msgs, max_new_tokens_cap // 2)
            for (j, flagged, _), text in zip(chunk, outs):
                try:
                    sub_map = parse_mark_map(text, len(flagged))
                except Exception:
                    continue
                for local_idx, orig_idx in enumerate(flagged, start=1):
                    if local_idx in sub_map:
                        pred_maps[j][orig_idx] = sub_map[local_idx]

    if predictions_json_path is not None:
        dump_predictions_json(predictions_json_path, raw_items, prop_items, pred_maps, raw_texts)

    if qwen_inputs_dir is not None:
        for idx, (raw, prop, text) in enumerate(zip(raw_items[:qwen_inputs_max], prop_items[:qwen_inputs_max], raw_texts[:qwen_inputs_max])):
            save_qwen_input_dump(qwen_inputs_dir / f"query_{idx:02d}_{Path(raw['path']).stem}", prompts, prop, icl, text)

    pairs: list[tuple[str, str]] = []
    for raw, prop, pred in zip(raw_items, prop_items, pred_maps):
        vlm_preds = [
            (prop["boxes"][j], pred.get(j + 1, "not_tomato"), 1.0)
            for j in range(len(prop["boxes"]))
            if pred.get(j + 1) in TOMATO_CLASSES
        ]
        gts = [(box, norm_label(lab)) for box, lab in zip(raw["boxes"], raw["labels"]) if norm_label(lab) in TOMATO_CLASSES]
        pairs.extend(match_det(gts, vlm_preds))
    return det_score(pairs, list(TOMATO_CLASSES))


def ckpt_eval(model, proc, prompts, test_items, proposals, icl, out_dir: Path, tag: str, step: int, best: list[float], extra: str = "", max_new_tokens_cap: int = 768) -> dict[str, Any]:
    """32장 eval 돌리고 `{tag}.log`/`{tag}_metrics.jsonl`에 남긴 뒤 best/last LoRA를 저장한다."""
    import torch

    torch.cuda.empty_cache()
    was_training = model.training
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    try:
        sc = eval_som_vlm(
            hf_generate_fn(model, proc), prompts, test_items, proposals, icl, bs=auto_bs(bool(icl)), desc=f"{tag}-{step}",
            predictions_json_path=out_dir / "predictions" / f"{tag}_step{step}.json",
            qwen_inputs_dir=out_dir / "qwen_inputs" / f"step_{step}", qwen_inputs_max=3,
            max_new_tokens_cap=max_new_tokens_cap,
        )
    finally:
        if was_training:
            model.train()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

    pc = sc["per_class"]
    line = (
        f"{tag} step={step} acc={sc['acc']:.3f} macroF1={sc['macro_f1']:.3f} "
        + " ".join(f"{c}={pc[c]['f1']:.3f}" for c in TOMATO_CLASSES)
        + (f" {extra}" if extra else "")
    ).rstrip()
    print(line, flush=True)
    with (out_dir / f"{tag}.log").open("a") as f:
        f.write(line + "\n")
    with (out_dir / f"{tag}_metrics.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "acc": sc["acc"], "macro_f1": sc["macro_f1"], "extra": extra}) + "\n")

    model.save_pretrained(out_dir / f"{tag}_lora_last")
    if sc["macro_f1"] >= best[0]:
        best[0] = sc["macro_f1"]
        model.save_pretrained(out_dir / f"{tag}_lora_best")
        print(f"  saved best macroF1={sc['macro_f1']:.3f}", flush=True)
    return sc


def encode_sft(proc, messages, device):
    from qwen_vl_utils import process_vision_info

    prompt_msgs = messages[:-1]
    prompt = _apply(proc, prompt_msgs, True)
    full = _apply(proc, messages, False)
    imgs, vids = process_vision_info(messages)
    full_in = proc(text=[full], images=imgs, videos=vids, return_tensors="pt")
    prompt_in = proc(text=[prompt], images=imgs, videos=vids, return_tensors="pt")
    labels = full_in["input_ids"].clone()
    labels[:, : prompt_in["input_ids"].shape[1]] = -100
    full_in["labels"] = labels
    return {k: v.to(device) for k, v in full_in.items()}
