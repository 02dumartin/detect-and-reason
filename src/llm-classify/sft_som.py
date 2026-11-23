"""SoM+mosaic VLM LoRA SFT — raw PyTorch 학습 루프 (ms-swift 미사용, llm-classify/sft_som.py 포팅).

ICL 1장 + staged VQA(dedup/filt/unripe/ripe_semi 중 랜덤 1개)로 매 step 학습한다.
20 step마다 32장 eval, best(macro-F1)/last LoRA를 저장한다.

Example:
    python "src/llm-classify/sft_som.py" --dataset little --config config/vlm_som/sft.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import autorootcwd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import som_common
import som_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoM VLM LoRA SFT (raw loop).")
    parser.add_argument("--dataset", type=str, required=True, help="config/dataset 아래의 데이터셋 이름 (little/tomatod/ag_rob_tomato)")
    parser.add_argument("--config", type=str, default="config/vlm_som/sft.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, help="config의 train.steps override")
    parser.add_argument(
        "--marker-style", choices=["box", "id_only", "box_center"], default="box",
        help="SoM 마크: box(테두리+모서리 번호, 기존) | id_only(테두리 없이 번호만) | box_center(테두리+중앙 번호)",
    )
    parser.add_argument("--exp-dir", type=str, default=None, help="exp/ 아래 실험 폴더명 override (기본: {오늘}_vlm-som)")
    parser.add_argument(
        "--image-adjustment", type=float, default=None,
        help="이미지 로드 시 적용할 밝기 배율 (예: 1.2=20%% 밝게, 0.8=20%% 어둡게). 기본 None=미적용.",
    )
    parser.add_argument(
        "--proposal-source", type=str, default="live",
        help='eval(test)에 쓸 proposal. "live"면 config의 YOLO를 그대로 씀. 모델 이름(예: rt_detr_1cls)이면 '
        "이미 predict된 결과를 재사용 — train은 항상 live YOLO proposal(노이즈 있는 걸로 학습해야 하므로 불변).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="물리 GPU 절대 인덱스. train proposal은 항상 live YOLO(collect_proposals)를 쓰는데, "
        "ultralytics가 predict 중 os.environ['CUDA_VISIBLE_DEVICES']를 이 값으로 덮어써버려서"
        "(2026-09-13 확인) 이 스크립트에서는 셸의 CUDA_VISIBLE_DEVICES 대신 이 옵션으로 GPU를 골라야 한다 "
        "— 셸에서 CUDA_VISIBLE_DEVICES를 같이 설정하면 충돌하니 쓰지 말 것.",
    )
    return parser.parse_args()


def make_train_msgs(prompts: som_common.SomPrompts, item: dict, exemplars: list[dict], rng: random.Random):
    kind, (q, a) = prompts.vqa_pair(item, rng)
    return prompts.with_icl(prompts.icl_stage(exemplars[:1], kind), item, text=q, answer=a)


def train_loop(model, proc, prompts, train_items, exemplars, test_items, proposals, icl, out_dir: Path, steps: int, lr: float, eval_every: int, max_new_tokens_cap: int) -> dict:
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best = [-1.0]
    score = None
    log_path = out_dir / "sft.log"
    with log_path.open("a") as log_file:
        for step in tqdm(range(steps), desc="sft"):
            rng = random.Random(step)
            item = som_common.shuffle_marks(train_items[step % len(train_items)], rng)
            loss = model(**som_common.encode_sft(proc, make_train_msgs(prompts, item, exemplars, rng), model.device)).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % eval_every == 0 or step + 1 == steps:
                message = f"step {step + 1} loss={loss.item():.4f}"
                print(message, flush=True)
                log_file.write(message + "\n")
                log_file.flush()
                score = som_common.ckpt_eval(
                    model, proc, prompts, test_items, proposals, icl, out_dir, "sft", step + 1, best,
                    extra=f"loss={loss.item():.4f}", max_new_tokens_cap=max_new_tokens_cap,
                )
    model.eval()
    return score


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["train"]
    steps = args.steps or train_cfg["steps"]
    # zero-shot(vLLM)은 backend config의 고정 max_new_tokens=768을 쓰는데 이쪽 hf 경로는
    # 마크수 비례 예산을 그대로 max_new_tokens로 쓰기 때문에, 상한이 너무 낮으면 마크 많은
    # 이미지에서 JSON이 중간에 잘려 그 이미지 전체 예측이 날아간다(2026-09-14 확인).
    # zero-shot과 같은 예산을 쓰도록 config로 노출한다.
    max_new_tokens_cap = cfg.get("generation", {}).get("max_new_tokens", 768)

    out_dir = som_dataset.default_output_dir(args.dataset, "sft", exp_dir=args.exp_dir)
    som_common.set_marker_style(args.marker_style)
    som_common.set_class_list(som_dataset.resolve_class_list(args.dataset))
    som_common.set_image_adjustment(args.image_adjustment)
    print(f"[sft_som] dataset={args.dataset} out_dir={out_dir}")

    train_raw = som_dataset.load_split(args.dataset, "train")
    if train_cfg.get("max_train"):
        train_raw = train_raw[: train_cfg["max_train"]]
    # best 체크포인트 선택은 val로, 최종 보고 점수는 학습 중 한 번도 안 본 test로 분리한다
    # (2026-09-14 -- 이전엔 test 32장을 선택에도 최종 평가에도 같이 써서 test-leakage였음).
    # max_eval: 0(기본)이면 val/test split 전체를 쓴다 -- 예전엔 32 고정이라 데이터셋
    # 크기와 무관하게 최대 32장만 보고 acc/macro_f1을 냈다(2026-09-14 확인, little 73장 중
    # 44%, custom_tomato 142장 중 22%만 반영됐었음).
    val_raw = som_dataset.load_split(args.dataset, "val")
    test_raw = som_dataset.load_split(args.dataset, "test")
    if train_cfg.get("max_eval"):
        val_raw = val_raw[: train_cfg["max_eval"]]
        test_raw = test_raw[: train_cfg["max_eval"]]

    prompts = som_common.SomPrompts(train_raw)
    (out_dir / "prompt.txt").write_text(prompts.full_prompt)

    proposal_cfg = cfg["proposal"]
    train_proposals = som_dataset.collect_proposals(
        args.dataset, train_raw, out_dir / "train_proposals.json",
        family=proposal_cfg["family"], conf=proposal_cfg["conf"], max_n=proposal_cfg["max_boxes"], device=args.gpu,
    )
    if args.proposal_source == "live":
        val_proposals = som_dataset.collect_proposals(
            args.dataset, val_raw, out_dir / "val_proposals.json",
            family=proposal_cfg["family"], conf=proposal_cfg["conf"], max_n=proposal_cfg["max_boxes"], device=args.gpu,
        )
        test_proposals = som_dataset.collect_proposals(
            args.dataset, test_raw, out_dir / "test_proposals.json",
            family=proposal_cfg["family"], conf=proposal_cfg["conf"], max_n=proposal_cfg["max_boxes"], device=args.gpu,
        )
    else:
        val_proposals = som_dataset.load_proposals_from_any_model_predictions(
            args.dataset, val_raw, model=args.proposal_source, split="val",
        )
        test_proposals = som_dataset.load_proposals_from_any_model_predictions(
            args.dataset, test_raw, model=args.proposal_source, split="test",
        )
    proposals = {**train_proposals, **val_proposals, **test_proposals}

    train_items = [
        som_common.label_props(it, proposals.get(it["path"], []), prompts.small_rel, prompts.small_side)
        for it in train_raw
    ]
    train_items = [it for it in train_items if it["boxes"]]
    oversample_max_factor = train_cfg.get("oversample_max_factor", 0)
    if oversample_max_factor:
        n_before = len(train_items)
        train_items = som_common.oversample_by_class(
            train_items, {r["path"]: r for r in train_raw}, max_factor=oversample_max_factor,
        )
        print(f"[sft_som] oversample_by_class: {n_before} -> {len(train_items)} items (max_factor={oversample_max_factor})")

    exemplars = som_common.pick_exemplars(
        train_raw,
        lambda it: som_common.label_props(it, proposals.get(it["path"], []), prompts.small_rel, prompts.small_side),
        n=cfg["icl"]["n_exemplars"],
    )
    icl = prompts.icl_messages(exemplars)
    stage_names = "dedup/filt/ripe_unripe" if len(som_common.TOMATO_CLASSES) == 2 else "dedup/filt/unripe/ripe_semi"
    print(f"train imgs={len(train_items)} proposal={proposal_cfg['family']}@{proposal_cfg['conf']} stages={stage_names}")

    model, proc = som_common.load_vlm(model_id=cfg["model_id"], device=args.device)
    lora_cfg = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
            task_type="CAUSAL_LM", target_modules=list(lora_cfg["target_modules"]),
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # best 체크포인트는 val로 고른다 (train_loop 내부의 주기적 ckpt_eval이 이걸로 판단).
    val_score = train_loop(model, proc, prompts, train_items, exemplars, val_raw, proposals, icl, out_dir, steps, train_cfg["lr"], train_cfg["eval_every"], max_new_tokens_cap)
    if val_score is None:
        val_score = som_common.eval_som_vlm(
            som_common.hf_generate_fn(model, proc), prompts, val_raw, proposals, icl,
            bs=som_common.auto_bs(bool(icl)), desc="sft-eval", max_new_tokens_cap=max_new_tokens_cap,
        )

    # 최종 보고 점수는 학습 중 선택에 한 번도 안 쓰인 test에서, best(val 기준) 체크포인트로
    # 딱 한 번만 잰다 (2026-09-14 -- 이전엔 test로 선택도 하고 그 test로 보고까지 해서
    # test-leakage였다. best_lora가 없으면(예: eval_every보다 steps가 짧음) 지금 메모리에
    # 있는 마지막 가중치로 대신 평가).
    best_lora_dir = out_dir / "sft_lora_best"
    if best_lora_dir.exists():
        model.load_adapter(str(best_lora_dir), adapter_name="best_for_test")
        model.set_adapter("best_for_test")
    score = som_common.eval_som_vlm(
        som_common.hf_generate_fn(model, proc), prompts, test_raw, proposals, icl,
        bs=som_common.auto_bs(bool(icl)), desc="sft-test", max_new_tokens_cap=max_new_tokens_cap,
    )

    (out_dir / "sft_result.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": args.dataset,
                "acc": score["acc"], "macro_f1": score["macro_f1"],
                "val_acc": val_score["acc"], "val_macro_f1": val_score["macro_f1"],
            },
            sort_keys=False,
        )
    )
    print(f"[sft_som] done test acc={score['acc']:.3f} macro_f1={score['macro_f1']:.3f} (val acc={val_score['acc']:.3f} macro_f1={val_score['macro_f1']:.3f})")


if __name__ == "__main__":
    main()
