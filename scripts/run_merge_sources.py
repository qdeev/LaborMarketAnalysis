"""Merge preprocessed source-level canonical tables."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import CanonicalSourceMerger, PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["separate", "combined", "experimental_merged"],
        default="separate",
        help=(
            "separate uses standalone Trudvsem/HH sources; combined uses only the "
            "external combined dataset; experimental_merged includes both."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing *_canonical.parquet source-level files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to data/processed/canonical_vacancies.parquet.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Output post-cleaning audit JSON path.",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    result = CanonicalSourceMerger(processed_dir=args.processed_dir).merge_sources(
        mode=args.mode,
        output_path=args.output,
        audit_path=args.audit,
        save=not args.no_save,
    )
    print(
        f"mode={result.mode}, rows={len(result.dataframe)}, "
        f"output={result.output_path}, audit={result.audit_path}"
    )


if __name__ == "__main__":
    main()
