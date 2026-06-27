"""Project paths and dataset definitions used by the website backend."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MONTHLY_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_broad_monthly_salary_dataset.parquet"
MONTHLY_MODEL_PATH = PROJECT_ROOT / "models" / "catboost_broad_monthly_residual_salary_model.cbm"
MONTHLY_SETUP_PATH = PROJECT_ROOT / "data" / "processed" / "broad_monthly_modeling_setup.json"
QUARTERLY_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_broad_quarterly_salary_dataset.parquet"
QUARTERLY_MODEL_PATH = PROJECT_ROOT / "models" / "catboost_broad_quarterly_residual_salary_model.cbm"
QUARTERLY_SETUP_PATH = PROJECT_ROOT / "data" / "processed" / "broad_quarterly_modeling_setup.json"


@dataclass(frozen=True)
class DatasetConfig:
    """Static schema and model settings for one aggregation horizon."""

    granularity: str
    dataset_path: Path
    model_path: Path
    setup_path: Path
    period_column: str
    target_period_column: str
    prediction_strategy: str


DATASET_CONFIGS = {
    "monthly": DatasetConfig(
        granularity="monthly",
        dataset_path=MONTHLY_DATASET_PATH,
        model_path=MONTHLY_MODEL_PATH,
        setup_path=MONTHLY_SETUP_PATH,
        period_column="vacancy_month",
        target_period_column="target_vacancy_month",
        prediction_strategy="monthly_residual_blend_0_7",
    ),
    "quarterly": DatasetConfig(
        granularity="quarterly",
        dataset_path=QUARTERLY_DATASET_PATH,
        model_path=QUARTERLY_MODEL_PATH,
        setup_path=QUARTERLY_SETUP_PATH,
        period_column="vacancy_quarter",
        target_period_column="target_vacancy_quarter",
        prediction_strategy="q4_to_q1_min_catboost_baseline",
    ),
}
