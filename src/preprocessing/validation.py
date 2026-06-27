"""Final result validation and audit helpers."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .aggregation import MONTHLY_SEGMENT_COLUMNS
from .canonical import CANONICAL_VACANCY_COLUMNS
from .constants import PROCESSED_DATA_DIR


@dataclass
class ValidationResult:
    """Validation report and output path."""

    report: dict[str, Any]
    output_path: Path | None = None


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("ResultValidator requires pandas.") from exc
    return pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _value_counts(series: Any) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _missing_columns(df: Any, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


class ResultValidator:
    """Validate final canonical vacancies and monthly segment outputs."""

    def __init__(self, processed_dir: str | Path = PROCESSED_DATA_DIR) -> None:
        self.processed_dir = Path(processed_dir)

    def validate_files(
        self,
        canonical_path: str | Path | None = None,
        monthly_path: str | Path | None = None,
        post_cleaning_audit_path: str | Path | None = None,
        source_audit_path: str | Path = "docs/data_audit.md",
        output_path: str | Path | None = None,
    ) -> ValidationResult:
        """Validate final parquet files and optionally save the audit report."""
        pd = _require_pandas()
        canonical_file = Path(canonical_path) if canonical_path else self.processed_dir / "canonical_vacancies.parquet"
        monthly_file = Path(monthly_path) if monthly_path else self.processed_dir / "canonical_monthly_segments.parquet"
        audit_file = (
            Path(post_cleaning_audit_path)
            if post_cleaning_audit_path
            else self.processed_dir / "canonical_vacancies_post_cleaning_audit.json"
        )

        if not canonical_file.exists():
            raise FileNotFoundError(f"Missing canonical vacancies parquet: {canonical_file}")
        if not monthly_file.exists():
            raise FileNotFoundError(f"Missing monthly segments parquet: {monthly_file}")

        canonical = pd.read_parquet(canonical_file)
        monthly = pd.read_parquet(monthly_file)
        report = self.validate_dataframes(
            canonical,
            monthly,
            post_cleaning_audit_path=audit_file,
            source_audit_path=source_audit_path,
        )
        report["files"] = {
            "canonical_vacancies": str(canonical_file),
            "canonical_monthly_segments": str(monthly_file),
            "post_cleaning_audit": str(audit_file),
            "source_audit": str(source_audit_path),
        }

        saved_path: Path | None = None
        if output_path is not None:
            saved_path = Path(output_path)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        return ValidationResult(report=report, output_path=saved_path)

    def validate_dataframes(
        self,
        canonical: Any,
        monthly: Any,
        post_cleaning_audit_path: str | Path | None = None,
        source_audit_path: str | Path = "docs/data_audit.md",
        date_success_threshold: float = 0.9,
    ) -> dict[str, Any]:
        """Validate already loaded final dataframes."""
        canonical_report = self.validate_canonical_vacancies(
            canonical,
            date_success_threshold=date_success_threshold,
        )
        monthly_report = self.validate_monthly_segments(monthly, canonical)
        audit_comparison = self.compare_post_cleaning_audit(
            canonical,
            post_cleaning_audit_path=post_cleaning_audit_path,
            source_audit_path=source_audit_path,
        )

        checks = [
            canonical_report["required_columns_present"],
            monthly_report["required_columns_present"],
            monthly_report["vacancy_count_matches_canonical_rows"],
            canonical_report["vacancy_date_parse_rate_ok"],
            canonical_report["salary_zero_check_passed"],
            audit_comparison["post_cleaning_audit_matches_current_rows"],
        ]
        return {
            "is_valid": all(checks),
            "canonical_vacancies": canonical_report,
            "canonical_monthly_segments": monthly_report,
            "audit_comparison": audit_comparison,
        }

    def validate_canonical_vacancies(
        self,
        df: Any,
        date_success_threshold: float = 0.9,
    ) -> dict[str, Any]:
        """Validate canonical vacancy-level output."""
        required_missing = _missing_columns(df, CANONICAL_VACANCY_COLUMNS)
        row_count = int(len(df))

        date_raw_count = int(df["vacancy_date_raw"].notna().sum()) if "vacancy_date_raw" in df.columns else 0
        date_parsed_count = int(df["vacancy_date"].notna().sum()) if "vacancy_date" in df.columns else 0
        date_parse_rate = date_parsed_count / date_raw_count if date_raw_count else 0.0

        salary_zero_counts = {}
        for column in ["salary_from", "salary_to", "salary_mid"]:
            if column in df.columns:
                numeric = _require_pandas().to_numeric(df[column], errors="coerce")
                salary_zero_counts[column] = int(numeric.eq(0).sum())
        salary_zero_check_passed = all(value == 0 for value in salary_zero_counts.values())

        duplicate_rows = int(df["is_duplicate"].sum()) if "is_duplicate" in df.columns else 0
        duplicate_groups = (
            int(df["duplicate_group_id"].nunique(dropna=True))
            if "duplicate_group_id" in df.columns
            else 0
        )

        return {
            "row_count": row_count,
            "required_missing_columns": required_missing,
            "required_columns_present": not required_missing,
            "vacancy_date_raw_count": date_raw_count,
            "vacancy_date_parsed_count": date_parsed_count,
            "vacancy_date_parse_rate": date_parse_rate,
            "vacancy_date_parse_rate_ok": date_parse_rate >= date_success_threshold,
            "salary_zero_counts": salary_zero_counts,
            "salary_zero_check_passed": salary_zero_check_passed,
            "salary_bound_type_distribution": _value_counts(df["salary_bound_type"])
            if "salary_bound_type" in df.columns
            else {},
            "salary_currency_distribution": _value_counts(df["salary_currency"])
            if "salary_currency" in df.columns
            else {},
            "source_dataset_distribution": _value_counts(df["source_dataset"])
            if "source_dataset" in df.columns
            else {},
            "duplicate_rows_marked": duplicate_rows,
            "duplicate_group_count": duplicate_groups,
        }

    def validate_monthly_segments(self, monthly: Any, canonical: Any | None = None) -> dict[str, Any]:
        """Validate canonical monthly segment output."""
        required_missing = _missing_columns(monthly, MONTHLY_SEGMENT_COLUMNS)
        vacancy_count_sum = (
            int(monthly["vacancy_count"].sum()) if "vacancy_count" in monthly.columns else 0
        )
        canonical_rows = int(len(canonical)) if canonical is not None else None

        return {
            "row_count": int(len(monthly)),
            "required_missing_columns": required_missing,
            "required_columns_present": not required_missing,
            "vacancy_count_sum": vacancy_count_sum,
            "canonical_row_count": canonical_rows,
            "vacancy_count_matches_canonical_rows": (
                vacancy_count_sum == canonical_rows if canonical_rows is not None else True
            ),
            "source_dataset_distribution": _value_counts(monthly["source_dataset"])
            if "source_dataset" in monthly.columns
            else {},
            "salary_missing_rate_min": float(monthly["salary_missing_rate"].min())
            if "salary_missing_rate" in monthly.columns and len(monthly)
            else None,
            "salary_missing_rate_max": float(monthly["salary_missing_rate"].max())
            if "salary_missing_rate" in monthly.columns and len(monthly)
            else None,
            "data_quality_score_min": float(monthly["data_quality_score"].min())
            if "data_quality_score" in monthly.columns and len(monthly)
            else None,
            "data_quality_score_max": float(monthly["data_quality_score"].max())
            if "data_quality_score" in monthly.columns and len(monthly)
            else None,
        }

    def compare_post_cleaning_audit(
        self,
        canonical: Any,
        post_cleaning_audit_path: str | Path | None = None,
        source_audit_path: str | Path = "docs/data_audit.md",
    ) -> dict[str, Any]:
        """Compare current canonical dataframe with saved post-cleaning and source audits."""
        audit_path = Path(post_cleaning_audit_path) if post_cleaning_audit_path else None
        source_path = Path(source_audit_path)
        audit_exists = bool(audit_path and audit_path.exists())
        source_audit_exists = source_path.exists()
        saved_row_count = None

        if audit_exists and audit_path is not None:
            saved = json.loads(audit_path.read_text(encoding="utf-8"))
            saved_row_count = saved.get("row_count")

        current_row_count = int(len(canonical))
        return {
            "source_audit_exists": source_audit_exists,
            "post_cleaning_audit_exists": audit_exists,
            "post_cleaning_audit_row_count": saved_row_count,
            "current_row_count": current_row_count,
            "post_cleaning_audit_matches_current_rows": (
                saved_row_count == current_row_count if audit_exists else True
            ),
        }
