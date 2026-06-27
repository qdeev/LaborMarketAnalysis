"""Validate final canonical vacancies and monthly segments outputs."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PROCESSED_DATA_DIR, ResultValidator  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--canonical",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_vacancies.parquet",
    )
    parser.add_argument(
        "--monthly",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
    )
    parser.add_argument(
        "--post-cleaning-audit",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_vacancies_post_cleaning_audit.json",
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=PROJECT_ROOT / "docs" / "data_audit.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "final_result_audit.json",
    )
    args = parser.parse_args()

    result = ResultValidator().validate_files(
        canonical_path=args.canonical,
        monthly_path=args.monthly,
        post_cleaning_audit_path=args.post_cleaning_audit,
        source_audit_path=args.source_audit,
        output_path=args.output,
    )
    print(f"is_valid={result.report['is_valid']}, output={result.output_path}")


if __name__ == "__main__":
    main()
