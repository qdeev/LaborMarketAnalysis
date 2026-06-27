"""Source-level preprocessing pipeline orchestration."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .canonical import CanonicalTransformer
from .constants import PROCESSED_DATA_DIR, RAW_DATASET_PATHS, SourceDataset, validate_source_dataset
from .dates import DateProcessor
from .io import DataProfiler, DatasetReader
from .normalization import Normalizer
from .quality import QualityChecker
from .salary import SalaryProcessor


@dataclass
class SourcePipelineResult:
    """Outputs produced by one source-level preprocessing run."""

    source_dataset: SourceDataset
    dataframe: Any
    raw_profile: dict[str, Any]
    validation_report: dict[str, Any]
    processing_report: dict[str, Any]
    output_path: Path | None
    report_path: Path | None


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("SourcePreprocessingPipeline requires pandas.") from exc
    return pd


def _records(frame: Any, limit: int = 10) -> list[dict[str, Any]]:
    if hasattr(frame, "empty") and not frame.empty:
        return frame.head(limit).where(frame.notna(), None).to_dict("records")
    return []


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class SourcePreprocessingPipeline:
    """Run the implemented preprocessing steps for one raw source dataset."""

    def __init__(
        self,
        reader: DatasetReader | None = None,
        profiler: DataProfiler | None = None,
        canonical: CanonicalTransformer | None = None,
        normalizer: Normalizer | None = None,
        dates: DateProcessor | None = None,
        salaries: SalaryProcessor | None = None,
        quality: QualityChecker | None = None,
    ) -> None:
        self.reader = reader or DatasetReader()
        self.profiler = profiler or DataProfiler()
        self.canonical = canonical or CanonicalTransformer()
        self.normalizer = normalizer or Normalizer()
        self.dates = dates or DateProcessor()
        self.salaries = salaries or SalaryProcessor()
        self.quality = quality or QualityChecker()

    def process_source(
        self,
        source_dataset: str,
        path: str | Path | None = None,
        read_kwargs: dict[str, Any] | None = None,
        output_dir: str | Path = PROCESSED_DATA_DIR,
        save: bool = True,
    ) -> SourcePipelineResult:
        """Process one raw source and optionally save parquet/report artifacts."""
        source = validate_source_dataset(source_dataset)
        dataset_path = Path(path) if path is not None else RAW_DATASET_PATHS[source]
        read_options = read_kwargs or {}

        raw = self.reader.read_dataset(dataset_path, source, **read_options)
        raw_profile = self.profiler.profile_dataframe(raw)
        before_rows = int(len(raw))

        df = self.canonical.to_canonical_schema(raw, source)
        df = self.normalizer.normalize_nulls(df)
        df = self.dates.select_vacancy_date(df, source)
        df = self.dates.extract_vacancy_month(df)
        df = self.dates.flag_impossible_dates(df, ["vacancy_date", "snapshot_date"])
        df = self.salaries.parse_salary(df, source)
        df = self.normalize_categories(df, source)
        df = self.quality.deduplicate_within_source(df, source)
        df = self.quality.add_quality_flags(df)

        validation_report = self.canonical.validate_canonical_schema(df)
        processing_report = self._build_processing_report(
            source=source,
            dataset_path=dataset_path,
            before_rows=before_rows,
            df=df,
            raw_profile=raw_profile,
            validation_report=validation_report,
        )

        output_path: Path | None = None
        report_path: Path | None = None
        if save:
            output_root = Path(output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            output_path = output_root / f"{source}_canonical.parquet"
            report_path = output_root / f"{source}_preprocessing_report.json"
            df.to_parquet(output_path, index=False)
            report_path.write_text(
                json.dumps(processing_report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

        return SourcePipelineResult(
            source_dataset=source,
            dataframe=df,
            raw_profile=raw_profile,
            validation_report=validation_report,
            processing_report=processing_report,
            output_path=output_path,
            report_path=report_path,
        )

    def normalize_categories(self, df: Any, source_dataset: str) -> Any:
        """Fill canonical category columns with normalized region, role, experience, and schedule."""
        pd = _require_pandas()
        out = df.copy()

        if "region" in out.columns:
            out["region_raw"] = out["region"]
            region_values = out["region"].map(self.normalizer.normalize_region)
            region_df = pd.DataFrame(region_values.tolist(), index=out.index)
            out["region"] = region_df["normalized"]
            out["region_is_unknown"] = region_df["is_unknown"]

        occupation_source = out["occupation_raw"] if "occupation_raw" in out.columns else out["title_raw"]
        occupation_values = occupation_source.map(
            lambda value: self.normalizer.normalize_occupation(value, source_dataset)
        )
        occupation_df = pd.DataFrame(occupation_values.tolist(), index=out.index)
        out["occupation_group"] = occupation_df["occupation_group"]
        out["title_normalized"] = out["title_raw"].where(out["title_raw"].notna(), occupation_df["normalized"])

        experience_values = out["experience_raw"].map(
            lambda value: self.normalizer.normalize_experience(value, source_dataset)
        )
        experience_df = pd.DataFrame(experience_values.tolist(), index=out.index)
        out["experience_min_years"] = experience_df["experience_min_years"]
        out["experience_max_years"] = experience_df["experience_max_years"]
        out["experience_group"] = experience_df["experience_group"]

        schedule_values = out["schedule_raw"].map(
            lambda value: self.normalizer.normalize_schedule(value, source_dataset)
        )
        schedule_df = pd.DataFrame(schedule_values.tolist(), index=out.index)
        out["schedule_type"] = schedule_df["schedule_type"]
        out["is_remote"] = schedule_df["is_remote"]
        out["is_shift"] = schedule_df["is_shift"]
        out["is_fly_in_fly_out"] = schedule_df["is_fly_in_fly_out"]
        return out

    def _build_processing_report(
        self,
        source: SourceDataset,
        dataset_path: Path,
        before_rows: int,
        df: Any,
        raw_profile: dict[str, Any],
        validation_report: dict[str, Any],
    ) -> dict[str, Any]:
        duplicate_count = int(df["is_duplicate"].sum()) if "is_duplicate" in df.columns else 0
        quality_flag_count = int(df["quality_flag_count"].sum()) if "quality_flag_count" in df.columns else 0

        return {
            "source_dataset": source,
            "source_file": str(dataset_path),
            "rows_before": before_rows,
            "rows_after": int(len(df)),
            "rows_lost": before_rows - int(len(df)),
            "duplicate_rows_marked": duplicate_count,
            "quality_flags_total": quality_flag_count,
            "validation": validation_report,
            "raw_profile": {
                "row_count": raw_profile["row_count"],
                "column_count": raw_profile["column_count"],
                "top_missing_values": _records(raw_profile["missing_values"], limit=10),
                "date_coverage": _records(raw_profile["date_coverage"], limit=20),
                "salary_columns": _records(raw_profile["salary_columns"], limit=20),
                "duplicates": _records(raw_profile["duplicate_summary"], limit=20),
            },
        }


def process_source_dataset(
    source_dataset: str,
    path: str | Path | None = None,
    read_kwargs: dict[str, Any] | None = None,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> SourcePipelineResult:
    """Convenience wrapper for processing one source dataset."""
    return SourcePreprocessingPipeline().process_source(
        source_dataset=source_dataset,
        path=path,
        read_kwargs=read_kwargs,
        output_dir=output_dir,
        save=save,
    )
