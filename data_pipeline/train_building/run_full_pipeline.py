#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full dataset-building pipeline: step1 -> step2 -> step3.")
    parser.add_argument("--raw-input", required=True, help="Raw input JSONL path.")
    parser.add_argument(
        "--work-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for intermediate outputs.",
    )
    parser.add_argument("--normalized-output", default="", help="Optional custom path for step1 output.")
    parser.add_argument("--inference-output", default="", help="Optional custom path for step2 output.")
    parser.add_argument("--final-output", default="", help="Optional custom path for final output.")

    parser.add_argument("--source-name", default="TimePLE")
    parser.add_argument("--data-type", default="grounding")
    parser.add_argument("--models", choices=("gemini", "qwen", "both"), default="both")
    parser.add_argument(
        "--step2-config",
        default="",
        help="YAML config path for step2_infer_models.py (required when running step2).",
    )
    parser.add_argument("--resume-inference", action="store_true")

    parser.add_argument("--skip-step1", action="store_true")
    parser.add_argument("--skip-step2", action="store_true")
    parser.add_argument("--skip-step3", action="store_true")

    parser.add_argument("--step1-extra-args", default="", help="Extra CLI args passed to step1 script.")
    parser.add_argument("--step2-extra-args", default="", help="Extra CLI args passed to step2 script.")
    parser.add_argument("--step3-extra-args", default="", help="Extra CLI args passed to step3 script.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(item) for item in command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized_output = Path(args.normalized_output) if args.normalized_output else work_dir / "step1_normalized.jsonl"
    inference_output = Path(args.inference_output) if args.inference_output else work_dir / "step2_with_predictions.jsonl"
    final_output = Path(args.final_output) if args.final_output else work_dir / "step3_with_iou.jsonl"

    step1_script = script_dir / "step1_normalize_dataset.py"
    step2_script = script_dir / "step2_infer_models.py"
    step3_script = script_dir / "step3_compute_iou.py"

    if not args.skip_step1:
        step1_cmd = [
            sys.executable,
            str(step1_script),
            "--input",
            str(args.raw_input),
            "--output",
            str(normalized_output),
            "--source-name",
            args.source_name,
            "--data-type",
            args.data_type,
        ]
        step1_cmd.extend(shlex.split(args.step1_extra_args))
        run_command(step1_cmd)

    if not args.skip_step2:
        if not args.step2_config:
            raise ValueError(
                "Step2 now runs in config-file mode. Please provide --step2-config "
                "(e.g., data_pipeline/train_building/infer_models_config.template.yaml)."
            )
        step2_cmd = [
            sys.executable,
            str(step2_script),
            "--config",
            str(args.step2_config),
            "--input",
            str(normalized_output),
            "--output",
            str(inference_output),
            "--models",
            args.models,
        ]
        if args.resume_inference:
            step2_cmd.append("--resume")
        step2_cmd.extend(shlex.split(args.step2_extra_args))
        run_command(step2_cmd)

    if not args.skip_step3:
        step3_cmd = [
            sys.executable,
            str(step3_script),
            "--input",
            str(inference_output),
            "--output",
            str(final_output),
        ]
        step3_cmd.extend(shlex.split(args.step3_extra_args))
        run_command(step3_cmd)

    print(
        f"[DONE] full pipeline finished.\n"
        f"step1={normalized_output}\n"
        f"step2={inference_output}\n"
        f"step3={final_output}"
    )


if __name__ == "__main__":
    main()
