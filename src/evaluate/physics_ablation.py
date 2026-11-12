"""Ablation of physics penalties on masked-quote IV surface reconstruction.

Trains the surface PINN under several penalty settings on identical data splits
and masks, over several seeds, and reports both accuracy (MAE/RMSE/R2 on hidden
quotes) and *physics violation* of the completed surface (RMS call-put IV gap,
RMS strike-curvature). A ``ground_truth`` row shows how much parity gap and
curvature the real surfaces themselves contain: a penalty is only sensible if
it does not push predictions far below that natural level.

Run from the repository root::

    python -m src.evaluate.physics_ablation --epochs 40
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.evaluate.masked_quote_benchmark import (
    _evaluate_classical,
    _fit_pinn,
    _fit_surface_mlp,
    _merged_base,
    _mlp_inputs,
    _prepare_surfaces,
    _residual_inputs,
    SPLIT_MODES,
    _score,
    load_physics_settings,
)
from src.models.surface_physics import (
    CURVATURE_MODES,
    SurfaceGeometry,
    surface_violation_metrics,
)

LOGGER = logging.getLogger(__name__)
METRIC_COLUMNS = ["mae", "rmse", "r2", "parity_rms", "curvature_rms"]


def ablation_variants(
    range_weight: float, parity_weight: float, curvature_weight: float
) -> Dict[str, Dict[str, float]]:
    """Penalty settings per PINN variant (weights of 0 switch a term off)."""
    return {
        "pinn_no_physics": dict(range_weight=0.0, parity_weight=0.0, curvature_weight=0.0),
        "pinn_range": dict(range_weight=range_weight, parity_weight=0.0, curvature_weight=0.0),
        "pinn_parity": dict(
            range_weight=range_weight, parity_weight=parity_weight, curvature_weight=0.0
        ),
        "pinn_curvature": dict(
            range_weight=range_weight, parity_weight=0.0, curvature_weight=curvature_weight
        ),
        "pinn_full": dict(
            range_weight=range_weight,
            parity_weight=parity_weight,
            curvature_weight=curvature_weight,
        ),
    }


def residual_variants(
    range_weight: float, parity_weight: float, curvature_weight: float
) -> Dict[str, Dict[str, float]]:
    """Residual-on-merged-interpolation variants: no penalties vs full penalties."""
    return {
        "pinn_residual": dict(
            range_weight=range_weight, parity_weight=0.0, curvature_weight=0.0
        ),
        "pinn_residual_full": dict(
            range_weight=range_weight,
            parity_weight=parity_weight,
            curvature_weight=curvature_weight,
        ),
    }


def _completed(values: np.ndarray, observed: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.where(observed, values, predictions)


def run_ablation(
    train_path: str,
    test_path: str,
    output_path: str,
    seeds: List[int],
    train_rows: int = 12000,
    evaluation_rows: int = 500,
    epochs: int = 40,
    range_weight: float = 0.1,
    parity_weight: float = 1.0,
    curvature_weight: float = 1.0,
    curvature_mode: str = "smooth",
    split_mode: str = "temporal",
    holdout_expiry: Optional[str] = None,
) -> pd.DataFrame:
    variants = ablation_variants(range_weight, parity_weight, curvature_weight)
    rows = []
    for seed in seeds:
        LOGGER.info("seed %d: preparing data", seed)
        data = _prepare_surfaces(
            train_path, test_path, train_rows, evaluation_rows, seed, split_mode, holdout_expiry
        )
        geometry = SurfaceGeometry.from_columns(data.strikes, data.call_side)
        splits = (
            ("validation", data.val_frame, data.val_values, data.val_masks,
             data.val_hidden, data.val_coordinates),
            ("holdout", data.test_frame, data.test_values, data.holdout_masks,
             data.holdout_hidden, data.test_coordinates),
        )

        models = {
            "surface_mlp": _fit_surface_mlp(
                data.train_values, data.train_coordinates, data.train_masks,
                epochs=epochs, seed=seed,
            )
        }
        for name, weights in variants.items():
            LOGGER.info("seed %d: training %s", seed, name)
            models[name] = _fit_pinn(
                data.train_values, data.train_coordinates, data.train_masks,
                epochs=epochs, seed=seed, curvature_mode=curvature_mode,
                geometry=geometry, **weights,
            )

        fill_value = float(np.nanmedian(data.train_values))
        residual_base = _merged_base(
            data.train_values, data.train_masks, data.strikes, fill_value
        )
        for name, weights in residual_variants(
            range_weight, parity_weight, curvature_weight
        ).items():
            LOGGER.info("seed %d: training %s", seed, name)
            models[name] = _fit_pinn(
                data.train_values, data.train_coordinates, data.train_masks,
                epochs=epochs, seed=seed, curvature_mode=curvature_mode,
                geometry=geometry, residual_base=residual_base, **weights,
            )

        for split, frame, values, masks, hidden, coordinates in splits:
            # reference rows: real surfaces and a classical baseline
            truth_metrics = surface_violation_metrics(values, geometry)
            rows.append(dict(seed=seed, split=split, method="ground_truth",
                             mae=np.nan, rmse=np.nan, r2=np.nan, **truth_metrics))
            for baseline in (
                "linear_interpolation", "parity_copy", "merged_interpolation",
                "merged_linear_wing", "merged_quad_wing",
            ):
                predicted, _ = _evaluate_classical(
                    baseline, frame, values, masks, data.strikes, data.call_side, hidden
                )
                rows.append(dict(
                    seed=seed, split=split, method=baseline,
                    **{k: v for k, v in _score(values, predicted, hidden).items()
                       if k in ("mae", "rmse", "r2")},
                    **surface_violation_metrics(_completed(values, masks, predicted), geometry),
                ))
            plain_input = _mlp_inputs(values, masks, coordinates).astype(np.float32)
            residual_input = _residual_inputs(
                values, masks, coordinates, data.strikes, fill_value
            )
            for name, model in models.items():
                model_input = residual_input if name.startswith("pinn_residual") else plain_input
                model.eval()
                with torch.no_grad():
                    predictions = model(torch.from_numpy(model_input)).numpy()
                scores = _score(values, predictions, hidden)
                rows.append(dict(
                    seed=seed, split=split, method=name,
                    **{k: scores[k] for k in ("mae", "rmse", "r2")},
                    **surface_violation_metrics(
                        _completed(values, masks, predictions), geometry
                    ),
                ))

    result = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = summarize(result)
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    return summary


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds for each split/method, in a stable method order."""
    grouped = result.groupby(["split", "method"], sort=False)[METRIC_COLUMNS]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/nk-iv-prediction/train_data.parquet")
    parser.add_argument("--test", default="data/nk-iv-prediction/test_data.parquet")
    parser.add_argument("--output", default="results/physics_ablation.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="temporal")
    parser.add_argument("--holdout-expiry", default=None)
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--evaluation-rows", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--range-weight", type=float, default=None)
    parser.add_argument("--parity-weight", type=float, default=None)
    parser.add_argument("--curvature-weight", type=float, default=None)
    parser.add_argument("--curvature-mode", choices=CURVATURE_MODES, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = load_physics_settings(args.config)
    ablation = settings["ablation"]
    seeds = args.seeds or list(ablation.get("seeds", [42]))
    pick = lambda cli, key, fallback: cli if cli is not None else ablation.get(key, fallback)
    summary = run_ablation(
        args.train,
        args.test,
        args.output,
        seeds=seeds,
        train_rows=args.train_rows,
        evaluation_rows=args.evaluation_rows,
        epochs=args.epochs,
        range_weight=args.range_weight if args.range_weight is not None else settings["range_weight"],
        parity_weight=pick(args.parity_weight, "parity_weight", 1.0),
        curvature_weight=pick(args.curvature_weight, "curvature_weight", 1.0),
        curvature_mode=args.curvature_mode or settings["curvature_mode"],
        split_mode=args.split_mode,
        holdout_expiry=args.holdout_expiry,
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        print(summary.to_string(index=False))
    print(f"\nSaved per-seed results to {args.output} and summary alongside it.")


if __name__ == "__main__":
    main()
