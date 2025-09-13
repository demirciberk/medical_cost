from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
import numpy as np

class MultivariateReview(ABC):
    @abstractmethod
    def review(self, data: pd.DataFrame) -> None:
        pass

class CorrelationReview(MultivariateReview):
    def review(self, data: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 8))
        corr = data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.show()
class PairPlotReview(MultivariateReview):
    def review(self, data: pd.DataFrame) -> None:
        sns.pairplot(data)
        plt.suptitle('Pair Plot', y=1.02)
        plt.show()
class OutlierReview(MultivariateReview):
    def review(self, data: pd.DataFrame) -> None:
         numerical_cols = data.select_dtypes(include=['number']).columns
         for col in numerical_cols:
             q1 = data[col].quantile(0.25)
             q3 = data[col].quantile(0.75)
             iqr = q3 - q1
             lower_bound = q1 - 1.5 * iqr
             upper_bound = q3 + 1.5 * iqr
             outliers = data[col][(data[col] < lower_bound) | (data[col] > upper_bound)]
             percentage = (len(outliers) / len(data[col])) * 100
             print(f"Outlier Percentage in {col}: {percentage:.2f}%")

class SetMultivariateStrategy:
    def __init__(self, strategy: MultivariateReview):
        self._strategy = strategy
    def set_strategy(self, strategy: MultivariateReview) -> None:
        self._strategy = strategy
    def execute_strategy(self, data: pd.DataFrame) -> None:
        self._strategy.review(data)
