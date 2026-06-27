"""DuckDB queries over prepared monthly and quarterly modelling datasets."""

from collections.abc import Iterable
from typing import Any

import duckdb

from app.config import DatasetConfig


FILTER_COLUMNS = ("source_dataset", "region", "occupation_group")
HH_SOURCE_DATASETS = ("hh_github", "hh_kaggle")


class AnalyticsRepository:
    """Read aggregated labour-market data without loading parquet into the API process."""

    def __init__(self, config: DatasetConfig) -> None:
        if not config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")
        self.config = config
        self.dataset_path = config.dataset_path
        self.period_column = config.period_column

    def filter_options(self, filters: dict[str, str | None]) -> dict[str, Any]:
        return {
            "granularity": self.config.granularity,
            "source_datasets": self._source_values(),
            "regions": self._values("region", filters, exclude="region"),
            "occupation_groups": self._values("occupation_group", filters, exclude="occupation_group"),
            "periods": self._values(self.period_column, filters),
        }

    def overview(self, filters: dict[str, str | None], period: str | None) -> dict[str, Any] | None:
        where_sql, params = self._where_clause(filters)
        selected_period = period or self._latest_period(where_sql, params)
        if selected_period is None:
            return None

        period_clause = self._append_clause(where_sql, f"{self.period_column} = ?")
        row = self._one(
            f"""
            SELECT
                COUNT(*) AS segment_count,
                median(median_salary_mid) AS median_salary_mid,
                CAST(sum(vacancy_count) AS BIGINT) AS vacancy_count,
                sum(salary_count) / nullif(sum(vacancy_count), 0) AS salary_coverage,
                avg(data_quality_score) AS data_quality_score
            FROM read_parquet(?)
            {period_clause}
            """,
            [str(self.dataset_path), *params, selected_period],
        )
        if row is None or row["segment_count"] == 0:
            return None
        return {"granularity": self.config.granularity, "period": selected_period, **row}

    def trend(self, filters: dict[str, str | None]) -> list[dict[str, Any]]:
        where_sql, params = self._where_clause(filters)
        return self._many(
            f"""
            SELECT
                {self.period_column} AS period,
                median(median_salary_mid) AS median_salary_mid,
                CAST(sum(vacancy_count) AS BIGINT) AS vacancy_count,
                sum(salary_count) / nullif(sum(vacancy_count), 0) AS salary_coverage
            FROM read_parquet(?)
            {where_sql}
            GROUP BY {self.period_column}
            ORDER BY {self.period_column}
            """,
            [str(self.dataset_path), *params],
        )

    def segment_rows(
        self,
        filters: dict[str, str | None],
        period: str | None,
        limit: int,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        where_sql, params = self._where_clause(filters)
        selected_period = period or self._latest_period(where_sql, params)
        if selected_period is None:
            return None, []

        period_clause = self._append_clause(where_sql, f"{self.period_column} = ?")
        rows = self._many(
            f"""
            SELECT
                source_dataset,
                region,
                occupation_group,
                {self.period_column} AS period,
                median_salary_mid,
                CAST(vacancy_count AS BIGINT) AS vacancy_count,
                CAST(salary_count AS BIGINT) AS salary_count
            FROM read_parquet(?)
            {period_clause}
            ORDER BY vacancy_count DESC, median_salary_mid DESC NULLS LAST
            LIMIT ?
            """,
            [str(self.dataset_path), *params, selected_period, limit],
        )
        return selected_period, rows

    def segment_history(
        self,
        source_dataset: str | None,
        region: str | None,
        occupation_group: str,
    ) -> list[dict[str, Any]]:
        where_sql, params = self._segment_where(source_dataset, region, occupation_group)
        return self._many(
            f"""
            SELECT
                {self.period_column} AS period,
                median(median_salary_mid) AS median_salary_mid,
                median(p25_salary_mid) AS p25_salary_mid,
                median(p75_salary_mid) AS p75_salary_mid,
                CAST(sum(vacancy_count) AS BIGINT) AS vacancy_count,
                CAST(sum(salary_count) AS BIGINT) AS salary_count,
                CAST(sum(employer_count) AS BIGINT) AS employer_count
            FROM read_parquet(?)
            {where_sql}
            GROUP BY {self.period_column}
            ORDER BY {self.period_column}
            """,
            [str(self.dataset_path), *params],
        )

    def forecast_row(
        self,
        source_dataset: str | None,
        region: str | None,
        occupation_group: str,
        period: str | None,
    ) -> dict[str, Any] | None:
        where_sql, params = self._segment_where(source_dataset, region, occupation_group)
        if period:
            period_clause = self._append_clause(where_sql, f"{self.period_column} = ?")
            period_params = [period]
        else:
            period_clause = where_sql
            period_params = []
        return self._one(
            f"""
            SELECT *
            FROM read_parquet(?)
            {period_clause}
            ORDER BY {self.period_column} DESC, vacancy_count DESC
            LIMIT 1
            """,
            [
                str(self.dataset_path),
                *params,
                *period_params,
            ],
        )

    def _segment_where(
        self,
        source_dataset: str | None,
        region: str | None,
        occupation_group: str,
    ) -> tuple[str, list[str]]:
        """Build the exact segment predicate, including HH rows without a region."""
        filters = {
            "source_dataset": source_dataset,
            "region": region,
            "occupation_group": occupation_group,
        }
        where_sql, params = self._where_clause(filters)
        if source_dataset == "hh" and not region:
            where_sql = self._append_clause(where_sql, "region IS NULL")
        return where_sql, params

    def _values(
        self,
        column: str,
        filters: dict[str, str | None],
        exclude: str | None = None,
    ) -> list[str]:
        effective_filters = {key: value for key, value in filters.items() if key != exclude}
        where_sql, params = self._where_clause(effective_filters)
        rows = self._many(
            f"""
            SELECT DISTINCT {column} AS value
            FROM read_parquet(?)
            {self._append_clause(where_sql, f'{column} IS NOT NULL')}
            ORDER BY value
            """,
            [str(self.dataset_path), *params],
        )
        return [row["value"] for row in rows]

    def _source_values(self) -> list[str]:
        values = set(self._values("source_dataset", {}))
        result: list[str] = []
        if "trudvsem_latest" in values:
            result.append("trudvsem_latest")
        if values.intersection(HH_SOURCE_DATASETS):
            result.append("hh")
        return result

    def _latest_period(self, where_sql: str, params: Iterable[Any]) -> str | None:
        row = self._one(
            f"SELECT max({self.period_column}) AS period FROM read_parquet(?) {where_sql}",
            [str(self.dataset_path), *params],
        )
        return None if row is None else row["period"]

    @staticmethod
    def _where_clause(filters: dict[str, str | None]) -> tuple[str, list[str]]:
        clauses: list[str] = []
        params: list[str] = []
        for column in FILTER_COLUMNS:
            value = filters.get(column)
            if value:
                if column == "source_dataset" and value == "hh":
                    clauses.append(f"{column} IN ({', '.join('?' for _ in HH_SOURCE_DATASETS)})")
                    params.extend(HH_SOURCE_DATASETS)
                else:
                    clauses.append(f"{column} = ?")
                    params.append(value)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params

    @staticmethod
    def _append_clause(where_sql: str, clause: str) -> str:
        return f"{where_sql} AND {clause}" if where_sql else f"WHERE {clause}"

    def _many(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        with duckdb.connect() as connection:
            return connection.execute(query, params).fetchdf().to_dict(orient="records")

    def _one(self, query: str, params: list[Any]) -> dict[str, Any] | None:
        rows = self._many(query, params)
        return rows[0] if rows else None
