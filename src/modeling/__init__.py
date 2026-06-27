"""Model training and evaluation utilities."""

from .catboost_training import (
    BroadMonthlyCatBoostTrainer,
    CatBoostTrainingResult,
    DEFAULT_CATBOOST_PARAMS,
)
from .backtesting import BroadMonthlyCatBoostBacktester, DEFAULT_BACKTEST_MONTHS
from .backtest_tuning import BroadMonthlyBacktestCatBoostTuner, DEFAULT_BACKTEST_TUNING_GRID
from .error_analysis import BroadMonthlyFoldErrorAnalyzer
from .ensemble import (
    BroadMonthlyEnsembleEvaluator,
    BroadQuarterlyEnsembleEvaluator,
    DEFAULT_ALPHA_GRID,
    EnsembleEvaluationResult,
)
from .evaluation import BroadMonthlyModelEvaluator
from .residual_backtesting import (
    BroadMonthlyResidualCatBoostBacktester,
    BroadQuarterlyResidualCatBoostBacktester,
    DEFAULT_BACKTEST_QUARTERS,
)
from .residual_training import (
    BroadMonthlyResidualCatBoostTrainer,
    BroadQuarterlyResidualCatBoostTrainer,
    RESIDUAL_PREDICTION_COLUMN,
    RESIDUAL_SALARY_PREDICTION_COLUMN,
    RESIDUAL_TARGET_COLUMN,
    ResidualCatBoostTrainingResult,
)
from .tuning import BroadMonthlyCatBoostTuner, DEFAULT_TUNING_GRID

__all__ = [
    "BroadMonthlyBacktestCatBoostTuner",
    "BroadMonthlyCatBoostBacktester",
    "BroadMonthlyEnsembleEvaluator",
    "BroadQuarterlyEnsembleEvaluator",
    "BroadMonthlyFoldErrorAnalyzer",
    "BroadMonthlyResidualCatBoostBacktester",
    "BroadMonthlyResidualCatBoostTrainer",
    "BroadQuarterlyResidualCatBoostBacktester",
    "BroadQuarterlyResidualCatBoostTrainer",
    "BroadMonthlyCatBoostTrainer",
    "BroadMonthlyCatBoostTuner",
    "BroadMonthlyModelEvaluator",
    "CatBoostTrainingResult",
    "EnsembleEvaluationResult",
    "DEFAULT_BACKTEST_MONTHS",
    "DEFAULT_BACKTEST_QUARTERS",
    "DEFAULT_BACKTEST_TUNING_GRID",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_CATBOOST_PARAMS",
    "DEFAULT_TUNING_GRID",
    "RESIDUAL_PREDICTION_COLUMN",
    "RESIDUAL_SALARY_PREDICTION_COLUMN",
    "RESIDUAL_TARGET_COLUMN",
    "ResidualCatBoostTrainingResult",
]
