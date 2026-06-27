"""CatBoost inference for the selected quarterly labour-market segment."""

import json
import math
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.config import DatasetConfig


class SalaryForecaster:
    """Apply a trained residual model with its horizon-specific strategy."""

    def __init__(self, config: DatasetConfig) -> None:
        if not config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {config.model_path}")
        if not config.setup_path.exists():
            raise FileNotFoundError(f"Model setup not found: {config.setup_path}")

        self.config = config
        self.setup = json.loads(config.setup_path.read_text(encoding="utf-8"))
        self.feature_columns: list[str] = self.setup["feature_columns"]
        self.categorical_features: list[str] = self.setup["catboost_cat_features"]
        self.model = CatBoostRegressor()
        self.model.load_model(str(config.model_path))

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        features = pd.DataFrame([{column: row[column] for column in self.feature_columns}])
        for column in self.categorical_features:
            features[column] = features[column].astype("string").fillna("__MISSING__")

        prediction_log_delta = float(
            self.model.predict(Pool(features, cat_features=self.categorical_features))[0]
        )
        baseline_salary = float(row["median_salary_mid"])
        catboost_salary = baseline_salary * math.exp(prediction_log_delta)
        q4_to_q1 = bool(row.get("is_q4_to_q1") == 1)
        if self.config.prediction_strategy == "monthly_residual_blend_0_7":
            forecast_salary = 0.7 * catboost_salary + 0.3 * baseline_salary
        elif q4_to_q1:
            forecast_salary = min(catboost_salary, baseline_salary)
        else:
            forecast_salary = catboost_salary

        return {
            "granularity": self.config.granularity,
            "source_dataset": str(row["source_dataset"]),
            "region": str(row["region"]),
            "occupation_group": str(row["occupation_group"]),
            "input_period": str(row[self.config.period_column]),
            "target_period": str(row[self.config.target_period_column]),
            "baseline_salary": baseline_salary,
            "catboost_salary": catboost_salary,
            "forecast_salary": forecast_salary,
            "predicted_change_percent": (forecast_salary / baseline_salary - 1) * 100,
            "q4_to_q1_correction_applied": q4_to_q1,
            "prediction_strategy": self.config.prediction_strategy,
        }
