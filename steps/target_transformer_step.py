import pandas as pd
from src.feature_engineering import FeatureEngineer, LogTransformation
from zenml import step


@step
def target_transformer_step(target: pd.Series) -> pd.Series:
    """
    Applies a log transformation to the target variable.

    Args:
        target: The pandas Series representing the target variable.

    Returns:
        A pandas Series with the log-transformed target variable.
    """
    # Convert Series to DataFrame to use the existing strategy
    target_df = target.to_frame()
    target_col_name = target.name

    # Use the LogTransformation strategy
    log_strategy = LogTransformation(features=[target_col_name])
    engineer = FeatureEngineer(log_strategy)

    # Apply the transformation
    transformed_df = engineer.apply_feature_engineering(target_df)

    # Convert back to Series
    return transformed_df[target_col_name]