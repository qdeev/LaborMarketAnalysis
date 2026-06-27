"""Audit monthly segments readiness for t+1 salary forecasting."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import MonthlyMLReadinessAuditor, PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "monthly_ml_readiness_audit.json",
    )
    args = parser.parse_args()

    result = MonthlyMLReadinessAuditor().audit_file(
        input_path=args.input,
        output_path=args.output,
    )
    report = result.report
    print(
        "monthly ML readiness: "
        f"ready={report.get('is_ready_for_next_step')}, "
        f"rows={report.get('row_count')}, "
        f"months={report.get('month_min')}..{report.get('month_max')}, "
        f"rows_with_t_plus_1={report.get('rows_with_next_month_target')}, "
        f"output={result.output_path}"
    )


if __name__ == "__main__":
    main()
