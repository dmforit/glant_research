from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def slug(value: Any) -> str:
    text = str(value).strip().replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-").lower() or "default"


def make_launch_id(dataset_names, mode: str, ablation_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    if isinstance(dataset_names, str):
        datasets_part = dataset_names
    else:
        datasets_part = "-".join(dataset_names)

    parts = [timestamp, slug(mode), slug(datasets_part)]

    if ablation_name:
        parts.append(slug(ablation_name))

    return "_".join(parts)


def make_run_dir(
    *,
    results_dir: str | Path,
    launch_id: str,
    dataset_name: str,
    model_name: str,
    ablation_name: str | None,
    seed: int,
    run_idx: int,
) -> Path:
    return (
        Path(results_dir)
        / "launches"
        / launch_id
        / "raw"
        / str(dataset_name)
        / str(model_name)
        / slug(ablation_name or "default")
        / f"seed_{int(seed)}"
        / f"run_{int(run_idx)}"
    )


def make_summary_dir(
    *,
    results_dir: str | Path,
    launch_id: str,
) -> Path:
    return Path(results_dir) / "launches" / launch_id / "summary"
