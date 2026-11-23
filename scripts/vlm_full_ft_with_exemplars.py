"""exemplar(+옵션 overlay) 기반 Qwen3-VL SFT 데이터 생성 + 학습 실행 CLI.

실제 로직은 src/vlm/dataset.py(crop/exemplar/overlay/jsonl 생성)와
src/vlm/finetuning.py(ms-swift SFT 커맨드 조립/실행/체크포인트 선택)에 있다.
이 스크립트는 argparse로 받은 값을 두 모듈의 config 객체로 변환해 넘기는 얇은 CLI다.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import output_name_for_rfdetr
from src.vlm.dataset import VlmDatasetConfig, build_experiment_suffix, build_vlm_dataset
from src.vlm.finetuning import FinetuneConfig, build_sft_command, run_sft_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-image exemplar SFT data and optionally run Qwen SFT."
    )
    parser.add_argument("--dataset-ref", default="merge")
    parser.add_argument("--class-mode", default="3cls")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=2,
        help="Number of exemplar crops per class to include in the prompt bundle.",
    )
    parser.add_argument("--expand-ratio", type=float, default=0.1)
    parser.add_argument(
        "--exemplar-expand-ratio",
        type=float,
        default=None,
        help="Crop expansion ratio for exemplar crops; defaults to --expand-ratio.",
    )
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument(
        "--exemplar-min-size",
        type=int,
        default=None,
        help="Minimum crop side length for exemplar crops; defaults to --min-size.",
    )
    parser.add_argument(
        "--exemplar-save-size",
        type=int,
        default=None,
        help="Optional square size to upscale exemplar crops before saving to manifest.",
    )
    parser.add_argument(
        "--exemplar-pick-strategy",
        default="brightness_then_area",
        choices=["brightness", "area", "area_then_brightness", "brightness_then_area", "class_aware"],
        help="How to choose exemplar crops within each class.",
    )
    parser.add_argument(
        "--crop-adjustment",
        default="none",
        choices=["none", "clahe"],
        help="Optional crop adjustment applied before saving training/eval crops.",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=0.03)
    parser.add_argument("--clahe-kernel-size", type=int, default=None)
    parser.add_argument(
        "--unripe-keep-ratio",
        type=float,
        default=1.0,
        help="Fraction of unripe train rows to keep. Validation/test are not downsampled.",
    )
    parser.add_argument(
        "--use-full-overlay",
        action="store_true",
        help=(
            "Add a full image with all detected boxes numbered plus a bbox coordinate "
            "list, in addition to the target crop, so the model also sees the object's "
            "spatial context."
        ),
    )
    parser.add_argument("--overlay-font-size", type=int, default=36)
    parser.add_argument("--overlay-box-thickness", type=int, default=3)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Seed for the unripe-class downsampling. Same seed -> same dataset.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-conversion", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--swift-cmd", default=os.environ.get("FULL_FT_SWIFT_CMD"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--tuner-type", default="full", choices=["full", "lora"])
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    return parser.parse_args()


def _build_dataset_config(args: argparse.Namespace) -> VlmDatasetConfig:
    crop_adjustment_cfg = None
    if args.crop_adjustment != "none":
        crop_adjustment_cfg = {"type": args.crop_adjustment}
        if args.crop_adjustment == "clahe":
            crop_adjustment_cfg["clip_limit"] = float(args.clahe_clip_limit)
            if args.clahe_kernel_size is not None:
                crop_adjustment_cfg["kernel_size"] = int(args.clahe_kernel_size)

    return VlmDatasetConfig(
        dataset_ref=args.dataset_ref,
        class_mode=args.class_mode,
        examples_per_class=args.examples_per_class,
        expand_ratio=args.expand_ratio,
        exemplar_expand_ratio=args.exemplar_expand_ratio,
        min_size=args.min_size,
        exemplar_min_size=args.exemplar_min_size,
        exemplar_save_size=args.exemplar_save_size,
        exemplar_pick_strategy=args.exemplar_pick_strategy,
        crop_adjustment_cfg=crop_adjustment_cfg,
        unripe_keep_ratio=args.unripe_keep_ratio,
        seed=args.sampling_seed,
        use_full_overlay=args.use_full_overlay,
        overlay_font_size=args.overlay_font_size,
        overlay_box_thickness=args.overlay_box_thickness,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )


def _build_finetune_config(args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        model_id=args.model_id,
        tuner_type=args.tuner_type,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        dataset_num_proc=args.dataset_num_proc,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        torch_dtype=args.torch_dtype,
        max_pixels=args.max_pixels,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        swift_cmd=args.swift_cmd,
        cuda_visible_devices=args.cuda_visible_devices,
    )


def main() -> None:
    args = parse_args()
    dataset_config = _build_dataset_config(args)
    finetune_config = _build_finetune_config(args)

    experiment_suffix = build_experiment_suffix(dataset_config)
    output_name = args.output_name or (
        f"{args.dataset_ref}_{args.class_mode}_qwen3_vl_full_ft_with_exemplars"
        f"_k{args.examples_per_class}{experiment_suffix}"
    )
    dataset_output_root = PROJECT_ROOT / "data" / "vlm_sft" / output_name
    train_jsonl = dataset_output_root / "train.jsonl"
    val_jsonl = dataset_output_root / "val.jsonl"
    run_dir_name = (
        "qwen3_vl_4b_decoder_lora"
        if args.tuner_type == "lora"
        else f"qwen3_vl_4b_full_ft_with_exemplars_k{args.examples_per_class}{experiment_suffix}"
    )
    train_output_dir = PROJECT_ROOT / "notebook" / "results" / "vlm_sft" / args.dataset_ref / run_dir_name
    coco_root = PROJECT_ROOT / "data" / "coco" / output_name_for_rfdetr(args.dataset_ref, args.class_mode)

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    if not args.skip_conversion:
        build_vlm_dataset(dataset_config, coco_root=coco_root, dataset_output_root=dataset_output_root)
    else:
        print("skip conversion")
        print(f"train jsonl = {train_jsonl}")
        print(f"val jsonl   = {val_jsonl}")
        if not train_jsonl.exists() or not val_jsonl.exists():
            raise FileNotFoundError("Existing train/val JSONL not found. Remove --skip-conversion first.")

    cmd = build_sft_command(
        finetune_config,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        output_dir=train_output_dir,
        project_root=PROJECT_ROOT,
    )
    print("\nTraining command:")
    print(shlex.join(cmd))

    if args.run_training:
        run_sft_training(
            cmd,
            project_root=PROJECT_ROOT,
            output_dir=train_output_dir,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    else:
        print("\nDry run only. Add --run-training to start training.")


if __name__ == "__main__":
    main()
