from __future__ import annotations

from copy import deepcopy
from typing import Any

from ml_collections import ConfigDict

from utils.model_names import canonical_model_name


GLANT_LAMBDA_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]


GLANT_ABLATIONS: list[dict[str, Any]] = [
    {
        "ablation_name": "gatv2_baseline",
        "model_name": "gatv2",
    },
    {
        "ablation_name": "glant_v1",
        "model_name": "glant_v1",
    },
    {
        "ablation_name": "glant_v2_lambda_0",
        "model_name": "glant_v2",
        "lambda_higher": 0.0,
    },
    {
        "ablation_name": "glant_v2_lambda_0_1",
        "model_name": "glant_v2",
        "lambda_higher": 0.1,
    },
    {
        "ablation_name": "glant_v2_lambda_0_25",
        "model_name": "glant_v2",
        "lambda_higher": 0.25,
    },
    {
        "ablation_name": "glant_v2_lambda_0_5",
        "model_name": "glant_v2",
        "lambda_higher": 0.5,
    },
    {
        "ablation_name": "glant_v2_lambda_1",
        "model_name": "glant_v2",
        "lambda_higher": 1.0,
    },
]


def ablation_names() -> list[str]:
    return [str(item["ablation_name"]) for item in GLANT_ABLATIONS]


def get_ablation(ablation_name: str) -> dict[str, Any]:
    for item in GLANT_ABLATIONS:
        if item["ablation_name"] == ablation_name:
            return deepcopy(item)

    available = ", ".join(ablation_names())
    raise ValueError(f"Unknown ablation_name={ablation_name!r}. Available: {available}")


def apply_ablation(config: ConfigDict, ablation_name: str) -> ConfigDict:
    """Apply one ablation preset to the global config in-place and return it."""
    preset = get_ablation(ablation_name)

    model_name = canonical_model_name(str(preset["model_name"]))
    config.ablation_name = str(preset["ablation_name"])
    config.baselines.names = [model_name]

    if model_name == "GLANT_v2":
        lambda_higher = float(preset["lambda_higher"])
        if not 0.0 <= lambda_higher <= 1.0:
            raise ValueError(f"lambda_higher must be in [0, 1], got {lambda_higher}")

        config.baselines.GLANT_v2.lambda_higher = lambda_higher
        config.baselines.GLANT_v2.learn_lambda_higher = False
        config.baselines.GLANT_v2.ablation_name = str(preset["ablation_name"])

    if model_name == "GLANT_v1":
        config.baselines.GLANT_v1.ablation_name = str(preset["ablation_name"])

    if model_name == "GATv2":
        config.baselines.GATv2.ablation_name = str(preset["ablation_name"])

    return config


def validate_glant_ablations() -> None:
    names = ablation_names()

    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate ablation names: {duplicates}")

    lambdas = sorted(
        float(item["lambda_higher"])
        for item in GLANT_ABLATIONS
        if canonical_model_name(str(item["model_name"])) == "GLANT_v2"
    )

    expected = sorted(GLANT_LAMBDA_VALUES)

    if lambdas != expected:
        raise ValueError(f"GLANT-v2 lambda values mismatch: got {lambdas}, expected {expected}")

    for item in GLANT_ABLATIONS:
        model_name = canonical_model_name(str(item["model_name"]))

        if model_name not in {"GATv2", "GLANT_v1", "GLANT_v2"}:
            raise ValueError(f"Invalid ablation model_name={item['model_name']!r}")

        if model_name == "GLANT_v2":
            if "lambda_higher" not in item:
                raise ValueError(f"{item['ablation_name']} missing lambda_higher")

            lambda_higher = float(item["lambda_higher"])
            if not 0.0 <= lambda_higher <= 1.0:
                raise ValueError(
                    f"{item['ablation_name']} has invalid lambda_higher={lambda_higher}"
                )
