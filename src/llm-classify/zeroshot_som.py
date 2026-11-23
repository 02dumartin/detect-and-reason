"""SoM+mosaic zero-shot / ICL 평가 (학습 없음, llm-classify/zeroshot_som.py 포팅).

proposal은 기본으로 학습된 1cls 검출기를 그 자리에서 낮은 conf로 돌려 만들지만
(`--proposal-source live`), 이미 predict를 돌려둔 모델의 COCO 예측을 그대로 재사용할
수도 있다(`--proposal-source <model_name>`, 예: rf_detr_1cls).

`--backend vllm`이면 HF `generate()` 대신 DaR이 이미 쓰고 있는
`src.vlm.backends.qwen_vllm.QwenVllmBackend`(in-process vLLM)로 추론한다 — Qwen3.5처럼
transformers 메인라인에 아직 없는 아키텍처(trust_remote_code 필요)를 돌릴 때 이 경로가 필요하다.
학습(SFT/RL)은 gradient가 필요해 vLLM으로는 못 하므로 이 옵션은 zero-shot/ICL 전용이다.

Example:
    python "src/llm-classify/zeroshot_som.py" --dataset ag_rob_tomato
    python "src/llm-classify/zeroshot_som.py" --dataset ag_rob_tomato --icl
    python "src/llm-classify/zeroshot_som.py" --dataset ag_rob_tomato --proposal-source rf_detr_1cls \
        --backend vllm --config config/vlm_som/zeroshot_qwen3_5_27b_vllm.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import autorootcwd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import som_common
import som_dataset

from src.evaluation.llm_classify_artifact_writer import save_llm_classify_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoM VLM zero-shot / ICL 평가 (학습 없음).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/vlm_som/sft.yaml", help="proposal/icl (+ vllm이면 backend/generation) 설정")
    parser.add_argument(
        "--max-eval", type=int, default=0,
        help="test 평가에 쓸 이미지 수 상한. 0(기본)이면 데이터셋의 test split 전체를 쓴다 "
        "-- 예전엔 32 고정이라 little(73장)/custom_tomato(142장)처럼 test가 32보다 큰 "
        "데이터셋은 일부만 보고 acc/macro_f1을 냈다(2026-09-14 확인).",
    )
    parser.add_argument("--icl", action="store_true", help="in-context example 2장을 붙인다")
    parser.add_argument(
        "--proposal-source",
        type=str,
        default="live",
        help='"live"면 1cls 검출기를 그 자리에서 돌림. 모델 이름(예: rf_detr_1cls)이면 이미 predict된 결과를 재사용.',
    )
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlay-max", type=int, default=None, help="predictions.json에서 렌더링할 [GT|예측] 오버레이 이미지 수 (기본: 전체)")
    parser.add_argument("--qwen-inputs-max", type=int, default=None, help="실제 VLM 입력(SoM/mosaic/prompt) 저장 장수 (기본: 전체 저장)")
    parser.add_argument(
        "--marker-style", choices=["box", "id_only", "box_center"], default="box",
        help="SoM 마크: box(테두리+모서리 번호, 기존) | id_only(테두리 없이 번호만) | box_center(테두리+중앙 번호)",
    )
    parser.add_argument("--exp-dir", type=str, default=None, help="exp/ 아래 실험 폴더명 override (기본: {오늘}_vlm-som)")
    parser.add_argument(
        "--image-adjustment", type=float, default=None,
        help="이미지 로드 시 적용할 밝기 배율 (예: 1.2=20%% 밝게, 0.8=20%% 어둡게). "
        "기본 None=미적용. 데이터셋별 조명 편향 보정용(som_common.set_image_adjustment).",
    )
    parser.add_argument("--exemplar", action="store_true", help="클래스별 참조 crop 이미지(색 보정용)를 프롬프트에 추가")
    parser.add_argument("--second-pass-review", action="store_true", help="1차 답변 전체를 스스로 재검토하는 2차 VLM 호출 추가(자기검증)")
    parser.add_argument("--recheck-not-tomato", action="store_true", help="1차에서 not_tomato로 나온 마크만 모아 재확인하는 VLM 호출 추가")
    return parser.parse_args()


def build_generate_fn(cfg: dict, backend: str, device: str):
    if backend == "hf":
        model, proc = som_common.load_vlm(model_id=cfg["model_id"], device=device)
        return som_common.hf_generate_fn(model, proc), som_common.auto_bs
    from src.vlm.backends.qwen_vllm import QwenVllmBackend

    vlm_backend = QwenVllmBackend(backend_cfg=cfg["backend"], generation_cfg=cfg["generation"])
    return som_common.vllm_generate_fn(vlm_backend), None


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    tag_parts = ["icl" if args.icl else "zeroshot"]
    if args.exemplar:
        tag_parts.append("exemplar")
    if args.second_pass_review:
        tag_parts.append("review")
    if args.recheck_not_tomato:
        tag_parts.append("recheck")
    tag = "_".join(tag_parts)
    out_dir = som_dataset.default_output_dir(args.dataset, tag, exp_dir=args.exp_dir)
    som_common.set_marker_style(args.marker_style)
    som_common.set_class_list(som_dataset.resolve_class_list(args.dataset))
    som_common.set_image_adjustment(args.image_adjustment)
    print(f"[zeroshot_som] dataset={args.dataset} icl={args.icl} backend={args.backend} proposal_source={args.proposal_source} out_dir={out_dir}")

    train_raw = som_dataset.load_split(args.dataset, "train")
    test_raw = som_dataset.load_split(args.dataset, "test")
    if args.max_eval:
        test_raw = test_raw[: args.max_eval]
    prompts = som_common.SomPrompts(train_raw)

    if args.proposal_source == "live":
        proposal_cfg = cfg["proposal"]
        proposals = som_dataset.collect_proposals(
            args.dataset, train_raw + test_raw, out_dir / "proposals.json",
            family=proposal_cfg["family"], conf=proposal_cfg["conf"], max_n=proposal_cfg["max_boxes"],
        )
    else:
        proposals = som_dataset.load_proposals_from_any_model_predictions(
            args.dataset, train_raw + test_raw, model=args.proposal_source, split="test",
        )

    icl_msgs = []
    if args.icl:
        # exemplar는 "정답 시연"이 목적이라 GT 박스/라벨을 그대로 쓴다(som_common.gt_labeled_item) —
        # proposal_source가 test split 예측만 갖고 있어(train 이미지엔 proposal이 없음,
        # 2026-09-11 확인) 검출기 proposal에 의존하면 라벨이 통째로 비었고, label_props의
        # too_small 필터까지 거치면 ag_rob_tomato처럼 GT 자체가 작은 데이터셋에서는 그마저도
        # 다 걸러졌다.
        exemplars = som_common.pick_exemplars(
            train_raw,
            som_common.gt_labeled_item,
            n=cfg["icl"]["n_exemplars"],
        )
        icl_msgs = prompts.icl_messages(exemplars)

    if args.exemplar:
        class_crops = som_common.pick_class_exemplar_crops(train_raw)
        icl_msgs = som_common.exemplar_preamble_messages(class_crops) + icl_msgs

    generate_fn, bs_fn = build_generate_fn(cfg, args.backend, args.device)
    bs = bs_fn(bool(icl_msgs)) if bs_fn else 8  # vLLM은 배치를 내부에서 알아서 스케줄링하므로 크게 잡아도 됨

    predictions_json_path = out_dir / "predictions.json"
    live_score = som_common.eval_som_vlm(
        generate_fn, prompts, test_raw, proposals, icl_msgs, bs=bs, desc=tag,
        predictions_json_path=predictions_json_path,
        qwen_inputs_dir=out_dir / "qwen_inputs", qwen_inputs_max=args.qwen_inputs_max,
        second_pass_review=args.second_pass_review, recheck_not_tomato=args.recheck_not_tomato,
    )

    # evaluation과 오버레이는 방금 저장한 predictions.json에서 다시 계산/렌더링한다 —
    # 모델을 다시 안 돌려도 이 파일 하나로 재현 가능하다는 걸 보장하기 위해 실시간 결과
    # (live_score)를 그대로 쓰지 않고 일부러 json을 다시 읽는다.
    score = som_common.evaluate_predictions_json(predictions_json_path)
    assert abs(score["acc"] - live_score["acc"]) < 1e-9, "predictions.json으로부터 재계산한 acc가 실시간 결과와 달라짐"

    evaluation_results = {
        "dataset": args.dataset, "backend": args.backend, "proposal_source": args.proposal_source,
        "predictions_json": str(predictions_json_path),
        "acc": score["acc"], "macro_f1": score["macro_f1"], "n_pairs": score["n_pairs"],
        "per_class": score["per_class"],
    }
    (out_dir / "evaluation_results.json").write_text(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

    # baseline VLM ablation(`vlm_classification_artifact_writer`)과 같은 형식으로
    # per-image/confusion까지 남긴다 — 위 acc/macro_f1과 값이 같아야 하므로 재계산 없이
    # predictions.json에서 다시 뽑는다(모델 재실행 불필요, 재현성 검증 동일 원칙).
    detailed_score = som_common.evaluate_predictions_json_detailed(predictions_json_path)
    detailed_score.update(
        dataset=args.dataset, backend=args.backend, proposal_source=args.proposal_source,
        predictions_json=str(predictions_json_path), iou_threshold=0.5,
    )
    save_llm_classify_artifacts(detailed_score, out_dir / "evaluation")

    n_overlays = som_common.render_overlays_from_predictions_json(
        predictions_json_path, out_dir / "overlays", max_images=args.overlay_max,
    )

    print(f"[zeroshot_som] done acc={score['acc']:.3f} macro_f1={score['macro_f1']:.3f} overlays={n_overlays}")
    for cls, stats in score["per_class"].items():
        print(f"  {cls}: P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")


if __name__ == "__main__":
    main()
