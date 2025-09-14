from typing import Tuple

import pandas as pd
from src.feature_engineering import (
    BinaryEncoder,
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Performs feature engineering using FeatureEngineer and the selected strategy.

    This step fits the transformer on the data and returns the transformed
    DataFrame along with the fitted FeatureEngineer object.

    Args:
        df: The input DataFrame to transform.
        strategy: The name of the strategy to use for feature engineering.
        features: The list of feature names to apply the strategy on.

    Returns:
        A tuple containing the transformed DataFrame and the fitted FeatureEngineer.
    """
    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer_strategy = LogTransformation(features)
    elif strategy == "standard_scaling":
        engineer_strategy = StandardScaling(features)
    elif strategy == "minmax_scaling":
        engineer_strategy = MinMaxScaling(features)
    elif strategy == "onehot_encoding":
        engineer_strategy = OneHotEncoding(features)
    elif strategy == "binary_encoding":
        engineer_strategy = BinaryEncoder(features)
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    engineer = FeatureEngineer(engineer_strategy)
    transformed_df = engineer.apply_feature_engineering(df, fit=True)
    return transformed_df, engineer