"""Merge source-level canonical vacancy tables."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
import warnings

from .constants import (
    PROCESSED_DATA_DIR,
    SOURCE_COMBINED,
    SOURCE_HH_GITHUB,
    SOURCE_HH_KAGGLE,
    SOURCE_TRUDVSEM_LATEST,
    SourceDataset,
)
from .quality import QualityChecker


BuildMode = Literal["separate", "combined", "experimental_merged"]

BUILD_MODE_SOURCES: dict[BuildMode, tuple[SourceDataset, ...]] = {
    "separate": (SOURCE_TRUDVSEM_LATEST, SOURCE_HH_GITHUB, SOURCE_HH_KAGGLE),
    "combined": (SOURCE_COMBINED,),
    "experimental_merged": (
        SOURCE_TRUDVSEM_LATEST,
        SOURCE_HH_GITHUB,
        SOURCE_HH_KAGGLE,
        SOURCE_COMBINED,
    ),
}


@dataclass
class MergeResult:
    """Outputs produced by canonical source merging."""

    mode: BuildMode
    dataframe: Any
    compatibility_report: dict[str, Any]
    audit_report: dict[str, Any]
    output_path: Path | None
    audit_path: Path | None


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("CanonicalSourceMerger requires pandas.") from exc
    return pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class CanonicalSourceMerger:
    """Combine preprocessed source-level canonical tables."""

    def __init__(
        self,
        processed_dir: str | Path = PROCESSED_DATA_DIR,
        quality: QualityChecker | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.quality = quality or QualityChecker()

    def source_path(self, source_dataset: SourceDataset) -> Path:
        """Return the default source-level parquet path for a dataset."""
        return self.processed_dir / f"{source_dataset}_canonical.parquet"

    def load_source_tables(
        self,
        sources: tuple[SourceDataset, ...],
        paths: dict[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        """Load source-level canonical parquet files."""
        pd = _require_pandas()
        tables: dict[str, Any] = {}
        explicit_paths = paths or {}

        for source in sources:
            path = Path(explicit_paths.get(source, self.source_path(source)))
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing source-level parquet for {source!r}: {path}. "
                    "Run scripts/run_source_pipeline.py first."
                )
            tables[source] = pd.read_parquet(path)
        return tables

    def merge_sources(
        self,
        mode: BuildMode = "separate",
        paths: dict[str, str | Path] | None = None,
        output_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        save: bool = True,
    ) -> MergeResult:
        """Merge source-level tables according to a conservative build mode."""
        pd = _require_pandas()
        if mode not in BUILD_MODE_SOURCES:
            raise ValueError(f"Unsupported build mode: {mode!r}")

        sources = BUILD_MODE_SOURCES[mode]
        tables = self.load_source_tables(sources, paths=paths)
        compatibility = self.quality.check_schema_compatibility(tables)
        if not compatibility["is_compatible"]:
            raise ValueError(
                "Source schemas are not merge-compatible: "
                + "; ".join(compatibility["errors"])
            )

        ordered_columns = list(next(iter(tables.values())).columns)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*DataFrame concatenation with empty or all-NA entries.*",
            )
            merged = pd.concat(
                [table.reindex(columns=ordered_columns) for table in tables.values()],
                ignore_index=True,
            )
        audit = self.build_post_cleaning_audit(merged, mode, compatibility)

        final_output_path: Path | None = None
        final_audit_path: Path | None = None
        if save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = (
                Path(output_path)
                if output_path is not None
                else self.processed_dir / "canonical_vacancies.parquet"
            )
            final_audit_path = (
                Path(audit_path)
                if audit_path is not None
                else self.processed_dir / "canonical_vacancies_post_cleaning_audit.json"
            )
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            final_audit_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(final_output_path, index=False)
            final_audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

        return MergeResult(
            mode=mode,
            dataframe=merged,
            compatibility_report=compatibility,
            audit_report=audit,
            output_path=final_output_path,
            audit_path=final_audit_path,
        )

    def build_post_cleaning_audit(
        self,
        df: Any,
        mode: BuildMode,
        compatibility_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a compact audit after source-level cleaning and merge."""
        source_counts = (
            df["source_dataset"].value_counts(dropna=False).sort_index().to_dict()
            if "source_dataset" in df.columns
            else {}
        )
        salary_bound_counts = (
            df["salary_bound_type"].value_counts(dropna=False).sort_index().to_dict()
            if "salary_bound_type" in df.columns
            else {}
        )
        currency_counts = (
            df["salary_currency"].value_counts(dropna=False).sort_index().to_dict()
            if "salary_currency" in df.columns
            else {}
        )
        quality_flag_counts: dict[str, int] = {}
        if "quality_flags" in df.columns:
            for flags in df["quality_flags"].dropna():
                for flag in flags if isinstance(flags, list) else []:
                    quality_flag_counts[flag] = quality_flag_counts.get(flag, 0) + 1

        return {
            "mode": mode,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "rows_by_source_dataset": {str(k): int(v) for k, v in source_counts.items()},
            "duplicate_rows_marked": int(df["is_duplicate"].sum())
            if "is_duplicate" in df.columns
            else 0,
            "vacancy_date_notna": int(df["vacancy_date"].notna().sum())
            if "vacancy_date" in df.columns
            else 0,
            "salary_bound_type_counts": {str(k): int(v) for k, v in salary_bound_counts.items()},
            "salary_currency_counts": {str(k): int(v) for k, v in currency_counts.items()},
            "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
            "schema_compatibility": compatibility_report,
            "combined_merge_warning": (
                "Mode experimental_merged includes the external combined dataset together "
                "with standalone sources. Use only for experiments until cross-source "
                "deduplication is validated."
            )
            if mode == "experimental_merged"
            else None,
        }


def merge_canonical_sources(
    mode: BuildMode = "separate",
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    save: bool = True,
    output_path: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> MergeResult:
    """Convenience wrapper for merging preprocessed source-level tables."""
    return CanonicalSourceMerger(processed_dir=processed_dir).merge_sources(
        mode=mode,
        output_path=output_path,
        audit_path=audit_path,
        save=save,
    )
