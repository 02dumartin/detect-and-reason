"""ms-swift 기반 Qwen VLM SFT(full fine-tuning/LoRA) 실행 오케스트레이션.

scripts/vlm_full_ft_with_exemplars.py가 갖고 있던 학습 커맨드 조립/실행/체크포인트
선택 로직을 분리한 모듈. 데이터 생성(src/vlm/dataset.py)과 책임을 나눠서, 학습
방식(하이퍼파라미터, tuner_type 등)을 바꿀 때 데이터 생성 코드를 건드리지 않게 한다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinetuneConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    tuner_type: str = "full"
    num_train_epochs: float = 2.0
    learning_rate: float | None = None
    train_batch_size: int | None = None
    eval_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    max_length: int = 1024
    dataset_num_proc: int = 4
    save_steps: int | None = None
    eval_steps: int | None = None
    logging_steps: int = 10
    save_total_limit: int = 2
    torch_dtype: str = "bfloat16"
    max_pixels: int | None = None
    deepspeed: str | None = None
    report_to: str = "tensorboard"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    swift_cmd: str | None = None
    cuda_visible_devices: str | None = None

    def with_resolved_defaults(self) -> "FinetuneConfig":
        """tuner_type(full/lora)에 따른 배치 크기 등 기본값을 채운 새 인스턴스를 반환한다."""
        is_lora = self.tuner_type == "lora"
        return FinetuneConfig(
            model_id=self.model_id,
            tuner_type=self.tuner_type,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate if self.learning_rate is not None else (1e-4 if is_lora else 1e-5),
            train_batch_size=self.train_batch_size if self.train_batch_size is not None else (2 if is_lora else 1),
            eval_batch_size=self.eval_batch_size if self.eval_batch_size is not None else (2 if is_lora else 1),
            gradient_accumulation_steps=(
                self.gradient_accumulation_steps if self.gradient_accumulation_steps is not None else (8 if is_lora else 16)
            ),
            max_length=self.max_length,
            dataset_num_proc=self.dataset_num_proc,
            save_steps=self.save_steps if self.save_steps is not None else (500 if is_lora else 200),
            eval_steps=self.eval_steps if self.eval_steps is not None else (500 if is_lora else 200),
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            torch_dtype=self.torch_dtype,
            max_pixels=self.max_pixels,
            deepspeed=self.deepspeed,
            report_to=self.report_to,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            swift_cmd=self.swift_cmd,
            cuda_visible_devices=self.cuda_visible_devices,
        )


def resolve_ms_swift_cmd(override: str | None, *, project_root: Path) -> str:
    candidates = [
        override,
        str(project_root / ".venv" / "bin" / "swift"),
        shutil.which("swift"),
    ]
    checked: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            checked.append(f"{candidate} (missing)")
            continue
        proc = subprocess.run(
            [str(path), "sft", "--help"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "openstack swift" not in text.lower() and ("usage: sft.py" in text or "swift/cli/sft.py" in text):
            return str(path)
        checked.append(f"{candidate} (not ms-swift)")
    raise RuntimeError("ModelScope ms-swift CLI not found. Checked: " + ", ".join(checked))


def build_sft_command(
    config: FinetuneConfig,
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    project_root: Path,
) -> list[str]:
    config = config.with_resolved_defaults()
    swift_cmd = resolve_ms_swift_cmd(config.swift_cmd, project_root=project_root)
    cmd = [
        swift_cmd,
        "sft",
        "--model",
        config.model_id,
        "--tuner_type",
        config.tuner_type,
        "--dataset",
        str(train_jsonl),
        "--val_dataset",
        str(val_jsonl),
        "--output_dir",
        str(output_dir),
        "--torch_dtype",
        config.torch_dtype,
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.train_batch_size),
        "--per_device_eval_batch_size",
        str(config.eval_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--learning_rate",
        str(config.learning_rate),
        "--max_length",
        str(config.max_length),
        "--dataset_num_proc",
        str(config.dataset_num_proc),
        "--save_steps",
        str(config.save_steps),
        "--eval_steps",
        str(config.eval_steps),
        "--logging_steps",
        str(config.logging_steps),
        "--save_total_limit",
        str(config.save_total_limit),
        "--use_hf",
        "true",
        "--report_to",
        config.report_to,
    ]
    if config.tuner_type == "lora":
        cmd.extend(
            [
                "--lora_rank",
                str(config.lora_rank),
                "--lora_alpha",
                str(config.lora_alpha),
                "--lora_dropout",
                str(config.lora_dropout),
                "--target_modules",
                config.target_modules,
                "--freeze_vit",
                "true",
                "--freeze_aligner",
                "true",
                "--freeze_llm",
                "false",
            ]
        )
    else:
        cmd.extend(
            [
                "--freeze_vit",
                "false",
                "--freeze_aligner",
                "false",
                "--freeze_llm",
                "false",
            ]
        )
    if config.max_pixels is not None:
        cmd.extend(["--max_pixels", str(config.max_pixels)])
    if config.deepspeed:
        cmd.extend(["--deepspeed", config.deepspeed])
    return cmd


def find_best_checkpoint(run_dir: Path) -> Path | None:
    trainer_states = sorted(
        run_dir.rglob("trainer_state.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for trainer_state in trainer_states:
        try:
            payload = json.loads(trainer_state.read_text(encoding="utf-8"))
        except Exception:
            continue
        checkpoint_ref = payload.get("best_model_checkpoint")
        if not checkpoint_ref:
            continue
        checkpoint_path = Path(str(checkpoint_ref))
        if checkpoint_path.exists():
            return checkpoint_path.resolve()
    return None


def publish_best_checkpoint_artifacts(run_dir: Path) -> None:
    best_checkpoint = find_best_checkpoint(run_dir)
    if best_checkpoint is None:
        print(f"[best-checkpoint] no best_model_checkpoint found under {run_dir}", flush=True)
        return

    best_txt = run_dir / "best_model_checkpoint.txt"
    best_txt.write_text(f"{best_checkpoint}\n", encoding="utf-8")

    best_link = run_dir / "best_model_checkpoint"
    try:
        if best_link.is_symlink() or best_link.exists():
            if best_link.is_symlink() or best_link.is_file():
                best_link.unlink()
        best_link.symlink_to(best_checkpoint, target_is_directory=True)
    except Exception as exc:
        print(f"[best-checkpoint] symlink creation skipped: {exc}", flush=True)

    print(f"[best-checkpoint] best={best_checkpoint}", flush=True)


def run_sft_training(
    cmd: list[str],
    *,
    project_root: Path,
    output_dir: Path,
    cuda_visible_devices: str | None,
) -> None:
    """swift sft 서브프로세스를 실행하고, 끝나면(성공/실패 무관) best checkpoint를 게시한다."""
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    print(f"\nRunning training with CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    try:
        subprocess.run(cmd, check=True, cwd=project_root, env=env)
    finally:
        publish_best_checkpoint_artifacts(output_dir)
