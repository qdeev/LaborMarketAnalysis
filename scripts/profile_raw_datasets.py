"""Generate a raw-data audit report for configured vacancy datasets."""

from argparse import ArgumentParser
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import profile_raw_datasets  # noqa: E402


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table_from_records(records: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in records:
        values = [_format_number(record.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _top_records(frame, sort_column: str, columns: list[str], limit: int = 10) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return (
        frame.sort_values(sort_column, ascending=False)
        .head(limit)[columns]
        .to_dict("records")
    )


def render_markdown(reports: dict[str, dict[str, Any]], sample_rows: int | None) -> str:
    scope = "full files" if sample_rows is None else f"first {sample_rows:,} rows"
    lines = [
        "# Generated Raw Dataset Audit",
        "",
        f"Scope: {scope}.",
        "",
    ]

    overview = [
        {
            "source_dataset": source,
            "row_count": report["row_count"],
            "column_count": report["column_count"],
        }
        for source, report in reports.items()
    ]
    lines.append(_table_from_records(overview, ["source_dataset", "row_count", "column_count"]))

    for source, report in reports.items():
        lines.extend(["", f"## {source}", ""])
        lines.append("Columns: `" + "`, `".join(report["columns"]) + "`")

        missing_rows = _top_records(
            report["missing_values"],
            "total_missing_rate",
            [
                "column",
                "nan_count",
                "empty_string_count",
                "placeholder_count",
                "total_missing_rate",
            ],
        )
        if missing_rows:
            lines.extend(["", "### Missing Values", ""])
            lines.append(
                _table_from_records(
                    missing_rows,
                    [
                        "column",
                        "nan_count",
                        "empty_string_count",
                        "placeholder_count",
                        "total_missing_rate",
                    ],
                )
            )

        date_rows = report["date_coverage"].to_dict("records")
        if date_rows:
            lines.extend(["", "### Dates", ""])
            lines.append(
                _table_from_records(
                    date_rows,
                    [
                        "column",
                        "non_empty_count",
                        "parsed_count",
                        "parse_success_rate",
                        "min",
                        "max",
                    ],
                )
            )

        salary_rows = report["salary_columns"].to_dict("records")
        if salary_rows:
            lines.extend(["", "### Salaries", ""])
            lines.append(
                _table_from_records(
                    salary_rows,
                    [
                        "column",
                        "non_empty_count",
                        "numeric_count",
                        "zero_count",
                        "min",
                        "max",
                        "median",
                        "potential_outlier_count",
                    ],
                )
            )

        duplicate_rows = report["duplicate_summary"].to_dict("records")
        if duplicate_rows:
            lines.extend(["", "### Duplicates", ""])
            lines.append(
                _table_from_records(
                    duplicate_rows,
                    ["scope", "duplicate_rows", "duplicate_groups", "duplicate_rate"],
                )
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "raw_data_audit_generated.md",
    )
    args = parser.parse_args()

    reports = profile_raw_datasets(sample_rows=args.sample_rows)
    markdown = render_markdown(reports, args.sample_rows)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote audit report: {args.output}")


if __name__ == "__main__":
    main()
