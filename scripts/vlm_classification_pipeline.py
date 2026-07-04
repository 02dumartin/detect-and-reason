from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import resolve_runtime_config
from src.vlm.classification import run_vlm_classification
from src.vlm.config import derive_output_dir, load_vlm_config
from src.vlm.util import (
    build_runtime_summary,
    emit_classification_mismatch_logs,
    fmt_metric,
    run_evaluation,
    run_family_export,
    run_overlay,
    save_classification_outputs,
    save_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM classification on detector predictions and save prediction/eval/overlay artifacts."
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Model config name under config/model")
    model_group.add_argument("--model-config", type=str, help="Explicit model config path")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="Dataset config name under config/dataset")
    dataset_group.add_argument("--dataset-config", type=str, help="Explicit dataset config path")

    parser.add_argument("--vlm-config", type=str, default="qwen3_vl_4b", help="VLM config name or path")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Target split override")
    parser.add_argument("--output-dir", type=str, help="Override output root. Default: model qwen_dir")
    parser.add_argument(
        "--detector-prediction-dir",
        type=str,
        help="Override detector prediction dir. Default: runtime prediction_dir",
    )
    parser.add_argument("--stdout-log", type=str, help="Write stdout progress logs to this file")
    parser.add_argument("--stderr-log", type=str, help="Write stderr warnings/errors to this file")
    parser.add_argument(
        "--stderr-only",
        action="store_true",
        help="Hide stdout from the terminal and keep only stderr visible there",
    )
    return parser.parse_args()


def main() -> None:
    started_at = time.time()
    args = parse_args()
    _configure_unbuffered_streams(args)
    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset

    print("[vlm_classification_pipeline] resolving runtime config", flush=True)
    runtime_cfg = resolve_runtime_config(
        model_ref=model_ref,
        dataset_ref=dataset_ref,
        stage="predict",
        source=args.split,
    )
    vlm_cfg, vlm_cfg_path = load_vlm_config(args.vlm_config, project_root=runtime_cfg["project_root"])

    print(f"[vlm_classification_pipeline] model={runtime_cfg['model_name']}", flush=True)
    print(f"[vlm_classification_pipeline] dataset={runtime_cfg['dataset']['key']}", flush=True)
    print(f"[vlm_classification_pipeline] split={args.split or vlm_cfg.get('input', {}).get('split') or 'test'}", flush=True)
    print(
        "[vlm_classification_pipeline] "
        f"detector_prediction_dir={args.detector_prediction_dir or runtime_cfg['paths']['prediction_dir']}",
        flush=True,
    )
    print(
        "[vlm_classification_pipeline] "
        f"output_dir={args.output_dir or runtime_cfg['paths'].get('qwen_dir') or '(derived)'}",
        flush=True,
    )
    print(
        "[vlm_classification_pipeline] "
        f"backend={vlm_cfg.get('backend', {}).get('type')} "
        f"model_name={vlm_cfg.get('backend', {}).get('model_name')} "
        f"batch_size={vlm_cfg.get('backend', {}).get('batch_size')}",
        flush=True,
    )
    print(
        "[vlm_classification_pipeline] "
        f"use_examples={bool(vlm_cfg.get('prompt', {}).get('use_examples'))} "
        f"example_picker={bool(vlm_cfg.get('prompt', {}).get('example_picker', {}).get('enabled'))}",
        flush=True,
    )

    output_dir = derive_output_dir(runtime_cfg, output_dir_override=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vlm.pipeline] output_dir={output_dir}", flush=True)

    classification_result = run_vlm_classification(
        runtime_cfg=runtime_cfg,
        vlm_cfg=vlm_cfg,
        split_override=args.split,
        detector_prediction_dir_override=args.detector_prediction_dir,
    )
    save_classification_outputs(output_dir=output_dir, classification_result=classification_result)

    predictions = classification_result["result_rows"]
    family_export: dict[str, object] = {}
    if vlm_cfg.get("output", {}).get("save_family_export", True):
        print("[vlm.pipeline] exporting_family_predictions", flush=True)
        family_export = run_family_export(
            runtime_cfg=runtime_cfg,
            predictions=predictions,
            output_dir=output_dir,
        )

    evaluation_output: dict[str, object] = {}
    if vlm_cfg.get("output", {}).get("save_eval", True):
        evaluation_output = run_evaluation(
            runtime_cfg=runtime_cfg,
            vlm_cfg=vlm_cfg,
            classification_result=classification_result,
            output_dir=output_dir,
        )
        metrics = evaluation_output.get("metrics", {})
        print(
            "[vlm.pipeline] "
            f"metrics detection_acc={fmt_metric(metrics.get('detection_acc'))} "
            f"classification_acc={fmt_metric(metrics.get('classification_acc'))} "
            f"overall_acc={fmt_metric(metrics.get('overall_acc'))}",
            flush=True,
        )
        emit_classification_mismatch_logs(evaluation_output.get("eval_rows", []))

    overlay_rendered = 0
    if vlm_cfg.get("output", {}).get("save_overlay", True):
        print("[vlm.pipeline] rendering_overlays", flush=True)
        overlay_rendered = run_overlay(
            runtime_cfg=runtime_cfg,
            vlm_cfg=vlm_cfg,
            predictions=predictions,
            output_dir=output_dir,
        )
        print(f"[vlm.pipeline] overlay_rendered={overlay_rendered}", flush=True)

    runtime_summary = build_runtime_summary(
        runtime_cfg=runtime_cfg,
        output_dir=output_dir,
        classification_result=classification_result,
        family_export=family_export,
        overlay_rendered=overlay_rendered,
        elapsed_sec=round(time.time() - started_at, 3),
    )
    save_run_metadata(
        output_dir=output_dir,
        runtime_cfg=runtime_cfg,
        vlm_cfg=vlm_cfg,
        runtime_summary=runtime_summary,
    )
    print(f"[vlm.pipeline] complete elapsed_sec={runtime_summary['elapsed_sec']}", flush=True)

    metrics = evaluation_output.get("metrics", {})
    print(f"[vlm_classification_pipeline] output_dir={output_dir}")
    print(f"[vlm_classification_pipeline] vlm_config={vlm_cfg_path or 'builtin-default'}")
    print(
        "[vlm_classification_pipeline] "
        f"detection_acc={fmt_metric(metrics.get('detection_acc'))} "
        f"classification_acc={fmt_metric(metrics.get('classification_acc'))} "
        f"overall_acc={fmt_metric(metrics.get('overall_acc'))}"
    )


def _configure_unbuffered_streams(args: argparse.Namespace) -> None:
    os.environ["PYTHONUNBUFFERED"] = "1"
    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)

    terminal_stdout = sys.stdout
    terminal_stderr = sys.stderr
    stdout_targets: list[TextIO] = []
    stderr_targets: list[TextIO] = [terminal_stderr]

    if not args.stderr_only:
        stdout_targets.append(terminal_stdout)
    if args.stdout_log:
        stdout_targets.append(_open_log_file(args.stdout_log))
    if args.stderr_log:
        stderr_targets.append(_open_log_file(args.stderr_log))
    if not stdout_targets:
        stdout_targets.append(_open_log_file(os.devnull))

    sys.stdout = _TeeStream(stdout_targets)
    sys.stderr = _TeeStream(stderr_targets)

    print(
        "[vlm_classification_pipeline] "
        f"logging stdout_log={args.stdout_log or 'disabled'} "
        f"stderr_log={args.stderr_log or 'disabled'} "
        f"stderr_only={args.stderr_only}",
        flush=True,
    )


def _reconfigure_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True, write_through=True)


def _open_log_file(path: str) -> TextIO:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8", buffering=1)


class _TeeStream:
    def __init__(self, streams: list[TextIO]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")
    
    def fileno(self) -> int:
        """Return file descriptor of the first stream that supports it."""
        for stream in self._streams:
            if hasattr(stream, 'fileno'):
                try:
                    return stream.fileno()
                except Exception:
                    continue
        raise AttributeError("No stream with fileno() available")
    


if __name__ == "__main__":
    main()
