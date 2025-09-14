import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): If True, fit and transform the data. If False, only transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.
        The 'fit' parameter is ignored as this is a stateless transformation.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): This parameter is ignored for log transformation.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): If True, fit and transform the data. If False, only transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        if fit:
            logging.info("Fitting and transforming data.")
            df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        else:
            logging.info("Transforming data using existing scaler.")
            df_transformed[self.features] = self.scaler.transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): If True, fit and transform the data. If False, only transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        if fit:
            logging.info("Fitting and transforming data.")
            df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        else:
            logging.info("Transforming data using existing scaler.")
            df_transformed[self.features] = self.scaler.transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for Binary Encoding
# -------------------------------------
# This strategy converts features with two distinct values into binary (0 and 1).
class BinaryEncoder(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the BinaryEncoder with the specific features to encode.

        Parameters:
        features (list): The list of features to apply binary encoding to.
        """
        self.features = features
        self.mappings = {}

    def apply_transformation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applies binary encoding to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): If True, fit and transform the data. If False, only transform.

        Returns:
        pd.DataFrame: The dataframe with binary encoded features.
        """
        logging.info(f"Applying binary encoding to features: {self.features}")
        df_transformed = df.copy()

        if fit:
            logging.info("Fitting and transforming data.")
            for feature in self.features:
                unique_values = sorted(df_transformed[feature].unique())
                if len(unique_values) != 2:
                    logging.warning(
                        f"Feature '{feature}' does not have exactly 2 unique values. Skipping."
                    )
                    continue
                self.mappings[feature] = {unique_values[0]: 0, unique_values[1]: 1}
                df_transformed[feature] = df_transformed[feature].map(
                    self.mappings[feature]
                )
        else:
            logging.info("Transforming data using existing mappings.")
            for feature in self.features:
                if feature in self.mappings:
                    df_transformed[feature] = df_transformed[feature].map(
                        self.mappings[feature]
                    )
                else:
                    logging.warning(
                        f"No mapping found for feature '{feature}'. Skipping transformation."
                    )

        logging.info("Binary encoding completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')

    def apply_transformation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): If True, fit and transform the data. If False, only transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()

        if fit:
            logging.info("Fitting and transforming data.")
            encoded_data = self.encoder.fit_transform(df[self.features])
        else:
            logging.info("Transforming data using existing encoder.")
            encoded_data = self.encoder.transform(df[self.features])

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.encoder.get_feature_names_out(self.features),
            index=df_transformed.index
        )
        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        fit (bool): If True, fit and transform the data. If False, only transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df, fit)


# Example usage
if __name__ == "__main__":
    # Example dataframes (e.g., train and test split)
    # df_train = pd.read_csv('../extracted-data/train_data.csv')
    # df_test = pd.read_csv('../extracted-data/test_data.csv')

    # Standard Scaling Example
    # scaler_features = ['age', 'bmi']
    # standard_scaler_strategy = StandardScaling(features=scaler_features)
    # scaler = FeatureEngineer(standard_scaler_strategy)
    
    # Fit on training data and transform it
    # df_train_scaled = scaler.apply_feature_engineering(df_train, fit=True)
    
    # Transform test data using the same fitted scaler
    # df_test_scaled = scaler.apply_feature_engineering(df_test, fit=False)

    # One-Hot Encoding Example
    # ohe_features = ['sex', 'smoker', 'region']
    # ohe_strategy = OneHotEncoding(features=ohe_features)
    # encoder = FeatureEngineer(ohe_strategy)

    # Fit on training data and transform it
    # df_train_encoded = encoder.apply_feature_engineering(df_train, fit=True)

    # Transform test data using the same fitted encoder
    # df_test_encoded = encoder.apply_feature_engineering(df_test, fit=False)

    # Binary Encoding Example
    # binary_features = ['sex'] # Assuming 'sex' has two values like 'male', 'female'
    # binary_strategy = BinaryEncoder(features=binary_features)
    # binary_encoder = FeatureEngineer(binary_strategy)

    # Fit on training data and transform it
    # df_train_binary = binary_encoder.apply_feature_engineering(df_train, fit=True)

    # Transform test data using the same fitted encoder
    # df_test_binary = binary_encoder.apply_feature_engineering(df_test, fit=False)

    pass