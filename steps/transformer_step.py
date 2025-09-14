import pandas as pd
from src.feature_engineering import FeatureEngineer
from zenml import step


@step
def transformer_step(
    df: pd.DataFrame, engineer: FeatureEngineer
) -> pd.DataFrame:
    """
    Applies a pre-fitted feature engineering transformation to a DataFrame.

    Args:
        df: The input DataFrame to transform.
        engineer: The fitted FeatureEngineer object.

    Returns:
        The transformed DataFrame.
    """
    transformed_df = engineer.apply_feature_engineering(df, fit=False)
    return transformed_df