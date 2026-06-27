"""Evaluate fallback/blending strategies for broad monthly predictions."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import BroadMonthlyEnsembleEvaluator  # noqa: E402
from src.preprocessing import PROCESSED_DATA_DIR  # noqa: E402


def _parse_alpha_grid(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROCESSED_DATA_DIR / "predictions_broad_monthly_residual_test.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_evaluation.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_metrics.csv",
    )
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    args = parser.parse_args()

    result = BroadMonthlyEnsembleEvaluator().evaluate_file(
        predictions_path=args.predictions,
        output_path=args.output,
        metrics_path=args.metrics,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
    )
    best = result.report["best_strategy"]
    print(
        "broad monthly ensemble evaluation complete: "
        f"best_strategy={best.get('strategy')}, "
        f"best_mape={best.get('mape'):.2f}%, "
        f"metrics={result.metrics_path}, "
        f"report={result.output_path}"
    )


if __name__ == "__main__":
    main()
