from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.ablation_config import ablation_names
from utils.model_names import canonical_model_name
from utils.run_paths import make_launch_id


DEFAULT_DATASETS = ["Cora", "Citeseer", "Texas", "AIFB", "IMDB", "ACM"]
DEFAULT_SEEDS = [0, 1, 2]

MODEL_GROUPS = {
    "run_baselines": ["gcn", "graphsage", "gatv2"],
    "run_khop_baselines": ["mixhop", "tagconv"],
    "run_hoga": ["hoga"],
    "run_glant_v1": ["glant_v1"],
    "run_glant_v2": ["glant_v2"],
}

MODE_CHOICES = [
    *MODEL_GROUPS,
    "run_ablation",
    "run_hpo",
    "collect_summary",
    "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for GLANT experiments.")
    parser.add_argument("--mode", choices=MODE_CHOICES, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-mode", choices=["hpo", "final", "baseline", "debug"], default="final")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--trials-v1", type=int, default=10)
    parser.add_argument("--trials-v2", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ablation", choices=ablation_names(), default=None)
    return parser.parse_args()


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(quote_cmd(cmd))
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def model_run_exists(
    *,
    results_dir: str,
    dataset: str,
    model: str,
    seed: int,
    ablation_name: str = "default",
) -> bool:
    canonical = canonical_model_name(model)
    launches_dir = Path(results_dir) / "launches"
    return bool(list(launches_dir.glob(
        f"*/raw/{dataset}/{canonical}/{ablation_name}/seed_{seed}/run_0/metrics.csv"
    )))


def hpo_exists(results_dir: str, launch_id: str) -> bool:
    return (
        Path(results_dir)
        / "launches"
        / launch_id
        / "summary"
        / "hpo_results.csv"
    ).exists()


def selected_modes(mode: str) -> list[str]:
    if mode == "all":
        return [
            "run_baselines",
            "run_khop_baselines",
            "run_hoga",
            "run_hpo",
            "run_glant_v1",
            "run_glant_v2",
            "run_ablation",
            "collect_summary",
        ]
    return [mode]


def selected_models(mode: str, explicit_models: list[str] | None) -> list[str]:
    if explicit_models is not None:
        return explicit_models
    return MODEL_GROUPS[mode]


def train_command(
    *,
    dataset: str,
    model: str,
    seed: int,
    args: argparse.Namespace,
    launch_id: str,
    run_mode: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--dataset",
        dataset,
        "--model",
        model,
        "--train",
        "--runs",
        str(args.runs),
        "--seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--results-dir",
        args.results_dir,
        "--run-mode",
        run_mode or args.run_mode,
        "--launch-id",
        launch_id,
    ]
    return cmd


def run_model_group(mode: str, args: argparse.Namespace, launch_id: str) -> None:
    models = selected_models(mode, args.models)
    for dataset in args.datasets:
        for model in models:
            for seed in args.seeds:
                if args.skip_existing and model_run_exists(
                    results_dir=args.results_dir,
                    dataset=dataset,
                    model=model,
                    seed=seed,
                ):
                    print(f"skip existing: {dataset} {model} seed={seed}")
                    continue
                run_command(
                    train_command(
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        args=args,
                        launch_id=launch_id,
                    ),
                    dry_run=args.dry_run,
                )


def run_ablation(args: argparse.Namespace, launch_id: str) -> None:
    selected = [args.ablation] if args.ablation else ablation_names()
    for dataset in args.datasets:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_glant_ablation.py"),
            "--dataset",
            dataset,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--runs",
            str(args.runs),
            "--gpu",
            str(args.gpu),
            "--results-dir",
            args.results_dir,
            "--run-mode",
            args.run_mode,
            "--launch-id",
            launch_id,
        ]
        if args.ablation:
            cmd.extend(["--ablation", args.ablation])
        else:
            cmd.append("--all")

        if args.dry_run:
            print(quote_cmd(cmd))
            continue

        if args.skip_existing:
            missing = False
            for ablation_name in selected:
                for seed in args.seeds:
                    if not model_run_exists(
                        results_dir=args.results_dir,
                        dataset=dataset,
                        model=_ablation_model(ablation_name),
                        seed=seed,
                        ablation_name=ablation_name,
                    ):
                        missing = True
                        break
                if missing:
                    break
            if not missing:
                print(f"skip existing ablation: {dataset}")
                continue

        subprocess.run(cmd, cwd=ROOT, check=True)


def _ablation_model(ablation_name: str) -> str:
    if ablation_name == "gatv2_baseline":
        return "gatv2"
    if ablation_name == "glant_v1":
        return "glant_v1"
    return "glant_v2"


def run_hpo(args: argparse.Namespace, launch_id: str) -> None:
    if args.skip_existing and hpo_exists(args.results_dir, launch_id):
        print(f"skip existing hpo: {launch_id}")
        return

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_glant_hpo.py"),
        "--datasets",
        *args.datasets,
        "--models",
        "glant_v1",
        "glant_v2",
        "--trials-v1",
        str(args.trials_v1),
        "--trials-v2",
        str(args.trials_v2),
        "--gpu",
        str(args.gpu),
        "--results-dir",
        args.results_dir,
        "--launch-id",
        launch_id,
    ]
    if args.trial_limit is not None:
        cmd.extend(["--trial-limit", str(args.trial_limit)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    run_command(cmd, dry_run=args.dry_run)


def collect_summary(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "collect_summary.py"),
        "--results-dir",
        args.results_dir,
    ]
    run_command(cmd, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    launch_id = args.launch_id or make_launch_id(
        dataset_names=args.datasets,
        mode=args.mode,
        ablation_name=None,
    )

    for mode in selected_modes(args.mode):
        if mode in MODEL_GROUPS:
            run_model_group(mode, args, launch_id)
        elif mode == "run_ablation":
            run_ablation(args, launch_id)
        elif mode == "run_hpo":
            run_hpo(args, launch_id)
        elif mode == "collect_summary":
            collect_summary(args)
        else:
            raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
