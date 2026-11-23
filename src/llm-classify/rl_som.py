"""SoM+mosaic VLM GRPO-lite RL — raw PyTorch 학습 루프 (llm-classify/rl_som.py 포팅).

sft_som.py가 저장한 LoRA(있으면)에서 이어서, 같은 VQA를 k번 샘플링해 advantage=r-r_mean
로 policy gradient + 0.5*SFT-CE 보조loss를 준다.

Example:
    python "src/llm-classify/rl_som.py" --dataset little --config config/vlm_som/rl.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import autorootcwd
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import som_common
import som_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoM VLM GRPO-lite RL (raw loop).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/vlm_som/rl.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, help="config의 train.steps override")
    parser.add_argument("--sft-lora", type=str, help="이어서 학습할 SFT LoRA 경로. 생략하면 같은 out_dir의 sft_lora_best/last를 찾는다")
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
        "이미 predict된 결과를 재사용 — train은 항상 live YOLO proposal(불변).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="물리 GPU 절대 인덱스. ultralytics(collect_proposals)가 CUDA_VISIBLE_DEVICES를 덮어써서"
        "(2026-09-13 확인) 셸의 CUDA_VISIBLE_DEVICES 대신 이 옵션으로 GPU를 골라야 한다 — 같이 쓰지 말 것.",
    )
    return parser.parse_args()


def resolve_k(free_gb: float, k_by_free_gb: list[dict]) -> int:
    for rule in k_by_free_gb:
        if free_gb >= rule["min_free_gb"]:
            return int(rule["k"])
    return 2


def sample_logprob(model, proc, prompts, item, exemplars, rng, temperature: float, kind: str | None = None):
    kind, (q, a) = prompts.vqa_pair(item, rng, kind=kind)
    icl = prompts.icl_stage(exemplars[:1], kind)
    model.eval()
    model.gradient_checkpointing_disable()
    try:
        text, inputs, out = som_common.generate(
            model, proc, prompts.with_icl(icl, item, text=q), max_new_tokens=192, do_sample=True, temperature=temperature
        )
    finally:
        model.train()
        model.gradient_checkpointing_enable()

    prompt_len = inputs["input_ids"].shape[1]
    labels = out.clone()
    labels[:, :prompt_len] = -100
    vision = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "labels")}
    if "mm_token_type_ids" in vision:
        # Qwen3.5(transformers>=5.17)의 3D RoPE(`get_rope_index`)가 mm_token_type_ids를
        # attention_mask와 같은 길이로 요구한다. 여기 `inputs`의 mm_token_type_ids는
        # prompt만 토크나이즈했을 때 길이(prompt_len)라, generate()가 이어붙인 생성 토큰
        # 만큼 짧다 -- 새로 생성된 토큰은 전부 텍스트이므로 0(텍스트 타입)으로 패딩해서
        # out과 길이를 맞춘다(2026-09-14 확인, 안 맞으면 IndexError로 죽음).
        mtt = vision["mm_token_type_ids"]
        pad_len = out.shape[1] - mtt.shape[1]
        if pad_len > 0:
            vision["mm_token_type_ids"] = torch.cat(
                [mtt, mtt.new_zeros(mtt.shape[0], pad_len)], dim=1
            )
    with torch.enable_grad():
        loss = model(input_ids=out, attention_mask=torch.ones_like(out), labels=labels, **vision).loss
    return text, -loss, q, a, kind


def train_loop(model, proc, prompts, train_items, exemplars, test_items, proposals, icl, out_dir: Path, steps: int, lr: float, eval_every: int, k: int, sft_ce_weight: float, temperature: float, max_new_tokens_cap: int) -> dict:
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best = [-1.0]
    score = None
    log_path = out_dir / "rl.log"
    with log_path.open("a") as log_file:
        for step in tqdm(range(steps), desc="rl"):
            rng = random.Random(step)
            item = som_common.shuffle_marks(train_items[step % len(train_items)], rng)
            # k개 rollout 전부 같은 stage(질문 종류)로 고정한다 -- 안 그러면 난이도가 다른
            # 질문들의 reward가 한 GRPO advantage 그룹(r.mean())에 섞인다(2026-09-14 확인).
            stage_kind = rng.choices(prompts.stage_names(), weights=prompts.stage_weights(), k=1)[0]

            logps, rewards = [], []
            q = a = kind = None
            for _ in range(k):
                text, logp, q, a, kind = sample_logprob(model, proc, prompts, item, exemplars, rng, temperature, kind=stage_kind)
                logps.append(logp)
                rewards.append(som_common.json_acc(a, text))

            r = torch.tensor(rewards, device=model.device, dtype=torch.float32)
            pg_loss = -((r - r.mean()).detach() * torch.stack(logps)).mean()
            try:
                ce_loss = model(
                    **som_common.encode_sft(proc, prompts.with_icl(prompts.icl_stage(exemplars[:1], kind), item, text=q, answer=a), model.device)
                ).loss
                loss = pg_loss + sft_ce_weight * ce_loss
            except IndexError as exc:
                # Qwen3.5(transformers==5.17.0)의 3D RoPE 위치 계산이 이 멀티턴+멀티이미지
                # ICL 입력에서 가끔 attention_mask/mm_token_type_ids 길이가 어긋나는 라이브러리
                # 버그를 냄(2026-09-14 확인, get_rope_index). 보조 CE-loss(가중치
                # sft_ce_weight)만 못 쓰는 거라, 그 step은 policy gradient loss만으로
                # 계속 진행한다 -- RL의 핵심 신호는 pg_loss라 학습 자체는 막히지 않는다.
                print(f"  [warn] step {step + 1}: CE-loss forward failed ({exc}); using pg_loss only", flush=True)
                loss = pg_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            extra = f"{kind} R={sum(rewards) / k:.3f} loss={loss.item():.4f}"
            if (step + 1) % eval_every == 0 or step + 1 == steps:
                message = f"step {step + 1} {extra}"
                print(message, flush=True)
                log_file.write(message + "\n")
                log_file.flush()
                score = som_common.ckpt_eval(model, proc, prompts, test_items, proposals, icl, out_dir, "rl", step + 1, best, extra=extra, max_new_tokens_cap=max_new_tokens_cap)
    model.eval()
    return score


def find_sft_lora(out_dir: Path, override: str | None) -> Path | None:
    if override:
        return Path(override)
    for name in ("sft_lora_best", "sft_lora_last"):
        candidate = out_dir.parent / "sft" / name
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["train"]
    steps = args.steps or train_cfg["steps"]
    # zero-shot(vLLM)과 같은 출력 예산을 쓰도록 config로 노출한다 (sft_som.py와 동일한 이유:
    # hf_generate_fn 경로는 마크수 비례 예산을 그대로 max_new_tokens로 써서, 상한이 너무
    # 낮으면 마크 많은 이미지에서 JSON이 잘려 그 이미지 전체 예측이 날아간다).
    max_new_tokens_cap = cfg.get("generation", {}).get("max_new_tokens", 768)

    out_dir = som_dataset.default_output_dir(args.dataset, "rl", exp_dir=args.exp_dir)
    som_common.set_marker_style(args.marker_style)
    som_common.set_class_list(som_dataset.resolve_class_list(args.dataset))
    som_common.set_image_adjustment(args.image_adjustment)
    print(f"[rl_som] dataset={args.dataset} out_dir={out_dir}")

    train_raw = som_dataset.load_split(args.dataset, "train")
    if train_cfg.get("max_train"):
        train_raw = train_raw[: train_cfg["max_train"]]
    # best 체크포인트 선택은 val, 최종 보고는 학습 중 안 쓰인 test로 분리 (2026-09-14,
    # sft_som.py와 동일한 이유 -- 이전엔 test로 선택도 보고도 같이 해서 leakage였음).
    # max_eval: 0(기본)이면 val/test split 전체를 쓴다 (sft_som.py와 동일한 이유,
    # 2026-09-14 확인).
    val_raw = som_dataset.load_split(args.dataset, "val")
    test_raw = som_dataset.load_split(args.dataset, "test")
    if train_cfg.get("max_eval"):
        val_raw = val_raw[: train_cfg["max_eval"]]
        test_raw = test_raw[: train_cfg["max_eval"]]

    prompts = som_common.SomPrompts(train_raw)

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
        it for it in (
            som_common.label_props(x, proposals.get(x["path"], []), prompts.small_rel, prompts.small_side)
            for x in train_raw
        )
        if it["boxes"]
    ]
    oversample_max_factor = train_cfg.get("oversample_max_factor", 0)
    if oversample_max_factor:
        n_before = len(train_items)
        train_items = som_common.oversample_by_class(
            train_items, {r["path"]: r for r in train_raw}, max_factor=oversample_max_factor,
        )
        print(f"[rl_som] oversample_by_class: {n_before} -> {len(train_items)} items (max_factor={oversample_max_factor})")

    exemplars = som_common.pick_exemplars(
        train_raw,
        lambda it: som_common.label_props(it, proposals.get(it["path"], []), prompts.small_rel, prompts.small_side),
        n=cfg["icl"]["n_exemplars"],
    )
    icl = prompts.icl_messages(exemplars)

    model, proc = som_common.load_vlm(model_id=cfg["model_id"], device=args.device)
    sft_lora = find_sft_lora(out_dir, args.sft_lora)
    lora_cfg = cfg["lora"]
    if sft_lora:
        print(f"[rl_som] resuming from SFT LoRA: {sft_lora}")
        model = PeftModel.from_pretrained(model, str(sft_lora), is_trainable=True)
    else:
        print("[rl_som] no SFT LoRA found — starting RL from a fresh LoRA")
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                task_type="CAUSAL_LM", target_modules=list(lora_cfg["target_modules"]),
            ),
        )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    k = resolve_k(free_gb, train_cfg["k_by_free_gb"])
    print(f"[rl_som] k={k} free={free_gb:.1f}GiB temperature={train_cfg['temperature']}")

    val_score = train_loop(
        model, proc, prompts, train_items, exemplars, val_raw, proposals, icl, out_dir,
        steps, train_cfg["lr"], train_cfg["eval_every"], k, train_cfg["sft_ce_weight"], train_cfg["temperature"],
        max_new_tokens_cap,
    )
    if val_score is None:
        val_score = som_common.eval_som_vlm(
            som_common.hf_generate_fn(model, proc), prompts, val_raw, proposals, icl,
            bs=som_common.auto_bs(bool(icl)), desc="rl-eval", max_new_tokens_cap=max_new_tokens_cap,
        )

    # 최종 보고는 val 기준 best 체크포인트를 학습 중 한 번도 안 본 test에서 한 번만 잰다.
    best_lora_dir = out_dir / "rl_lora_best"
    if best_lora_dir.exists():
        model.load_adapter(str(best_lora_dir), adapter_name="best_for_test")
        model.set_adapter("best_for_test")
    score = som_common.eval_som_vlm(
        som_common.hf_generate_fn(model, proc), prompts, test_raw, proposals, icl,
        bs=som_common.auto_bs(bool(icl)), desc="rl-test", max_new_tokens_cap=max_new_tokens_cap,
    )

    (out_dir / "rl_result.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": args.dataset,
                "acc": score["acc"], "macro_f1": score["macro_f1"],
                "val_acc": val_score["acc"], "val_macro_f1": val_score["macro_f1"],
            },
            sort_keys=False,
        )
    )
    print(f"[rl_som] done test acc={score['acc']:.3f} macro_f1={score['macro_f1']:.3f} (val acc={val_score['acc']:.3f} macro_f1={val_score['macro_f1']:.3f})")


if __name__ == "__main__":
    main()
