from abc import ABC, abstractmethod
import pandas as pd

class DataReview(ABC):
    @abstractmethod
    def review(self, data: pd.DataFrame) -> None:
        pass
class InfoReview(DataReview):
    def review(self, data: pd.DataFrame) -> None:
        print("Data Info:")
        print(data.info())
class DescribeReview(DataReview):
    def review(self, data: pd.DataFrame) -> None:
        print("Data Description (Numerical Columns):")
        print(data.describe())
        print("\nData Description (Categorical Columns):")
        print(data.describe(include=['object', 'category']))
class MissingValuesReview(DataReview):
    def review(self, data: pd.DataFrame) -> None:
        print("Missing Values:")
        missing_values = data.isnull().sum()
        print(missing_values[missing_values > 0])

class SetStrategy:
    def __init__(self,strategy: DataReview):
        self._strategy = strategy
    def set_strategy(self, strategy: DataReview) -> None:
        self._strategy = strategy
    def execute_strategy(self, data: pd.DataFrame) -> None:
        self._strategy.review(data)
