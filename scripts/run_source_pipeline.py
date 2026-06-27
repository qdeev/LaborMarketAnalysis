"""Run source-level preprocessing for one or more raw datasets."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    PROCESSED_DATA_DIR,
    SUPPORTED_SOURCE_DATASETS,
    SourcePreprocessingPipeline,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        choices=[*SUPPORTED_SOURCE_DATASETS, "all"],
        default="all",
        help="Source dataset to process.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only the first N rows for a quick pipeline run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for parquet outputs and JSON reports.",
    )
    parser.add_argument("--no-save", action="store_true", help="Run without writing artifacts.")
    args = parser.parse_args()

    sources = SUPPORTED_SOURCE_DATASETS if args.source == "all" else (args.source,)
    pipeline = SourcePreprocessingPipeline()

    for source_dataset in sources:
        read_kwargs = {}
        if args.sample_rows is not None:
            read_kwargs = (
                {"limit": args.sample_rows}
                if source_dataset == "hh_github"
                else {"nrows": args.sample_rows}
            )

        result = pipeline.process_source(
            source_dataset,
            read_kwargs=read_kwargs,
            output_dir=args.output_dir,
            save=not args.no_save,
        )
        print(
            f"{source_dataset}: rows_before={result.processing_report['rows_before']}, "
            f"rows_after={result.processing_report['rows_after']}, "
            f"duplicates={result.processing_report['duplicate_rows_marked']}, "
            f"output={result.output_path}"
        )


if __name__ == "__main__":
    main()
