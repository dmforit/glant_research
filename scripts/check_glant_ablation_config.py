from __future__ import annotations

import sys
from pathlib import Path

import torch
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.ablation_config import (
    GLANT_ABLATIONS,
    GLANT_LAMBDA_VALUES,
    apply_ablation,
    validate_glant_ablations,
)
from configs.config import all_config
from utils.model_names import canonical_model_name
from utils.model_utils import create_model


def make_ds_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.in_channels = 5
    cfg.out_channels = 3
    cfg.num_classes = 3
    cfg.num_nodes = 12
    cfg.name = "SyntheticAblationCheck"
    return cfg


def check_presets() -> None:
    validate_glant_ablations()

    names = [item["ablation_name"] for item in GLANT_ABLATIONS]
    if len(names) != len(set(names)):
        raise AssertionError("Ablation names are not unique")

    lambdas = sorted(
        float(item["lambda_higher"])
        for item in GLANT_ABLATIONS
        if canonical_model_name(str(item["model_name"])) == "GLANT_v2"
    )
    expected = sorted(GLANT_LAMBDA_VALUES)

    if lambdas != expected:
        raise AssertionError(f"lambda list mismatch: got {lambdas}, expected {expected}")

    print("ok ablation presets")


def check_apply_and_create_models() -> None:
    ds_config = make_ds_config()

    for preset in GLANT_ABLATIONS:
        ablation_name = str(preset["ablation_name"])
        config = all_config()
        config.device = torch.device("cpu")

        apply_ablation(config, ablation_name)

        if getattr(config, "ablation_name") != ablation_name:
            raise AssertionError(f"{ablation_name}: config.ablation_name was not set")

        if len(config.baselines.names) != 1:
            raise AssertionError(f"{ablation_name}: expected exactly one selected model")

        model_name = canonical_model_name(config.baselines.names[0])
        expected_model = canonical_model_name(str(preset["model_name"]))

        if model_name != expected_model:
            raise AssertionError(
                f"{ablation_name}: selected model {model_name}, expected {expected_model}"
            )

        if model_name == "GLANT_v2":
            expected_lambda = float(preset["lambda_higher"])
            actual_lambda = float(config.baselines.GLANT_v2.lambda_higher)

            if actual_lambda != expected_lambda:
                raise AssertionError(
                    f"{ablation_name}: lambda_higher={actual_lambda}, expected={expected_lambda}"
                )

        model = create_model(model_name, config, ds_config)
        if model is None:
            raise AssertionError(f"{ablation_name}: create_model returned None")

    print("ok apply_ablation and model factory")


def main() -> None:
    check_presets()
    check_apply_and_create_models()
    print("All GLANT ablation config checks passed.")


if __name__ == "__main__":
    main()
