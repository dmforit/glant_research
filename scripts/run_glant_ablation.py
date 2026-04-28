from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.ablation_config import GLANT_ABLATIONS, ablation_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ablation", type=str, default=None, choices=ablation_names())
    parser.add_argument("--all", action="store_true", help="Run all ablation presets")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--run-mode", type=str, default="final", choices=["hpo", "final", "baseline", "debug"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch-id", type=str, default=None)

    args = parser.parse_args()

    if args.all == (args.ablation is not None):
        raise ValueError("Specify exactly one of --all or --ablation")

    return args


def selected_ablations(args: argparse.Namespace) -> list[str]:
    if args.all:
        return [str(item["ablation_name"]) for item in GLANT_ABLATIONS]

    return [str(args.ablation)]


def build_command(args: argparse.Namespace, ablation_name: str, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--dataset",
        args.dataset,
        "--train",
        "--ablation",
        ablation_name,
        "--seed",
        str(seed),
        "--runs",
        str(args.runs),
        "--run-mode",
        args.run_mode,
    ]

    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])

    if args.results_dir is not None:
        cmd.extend(["--results-dir", args.results_dir])

    if args.launch_id is not None:
        cmd.extend(["--launch-id", args.launch_id])

    return cmd


def main() -> None:
    args = parse_args()

    for ablation_name in selected_ablations(args):
        for seed in args.seeds:
            cmd = build_command(args, ablation_name, seed)

            if args.dry_run:
                print(" ".join(cmd))
                continue

            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
