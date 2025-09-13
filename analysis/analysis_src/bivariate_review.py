from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateReview(ABC):
    @abstractmethod
    def review(self, data: pd.DataFrame, column1: str, column2: str) -> None:
        pass

class Numerical_NumericalReview(BivariateReview):
    def review(self, data, column1, column2):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=data[column1], y=data[column2])
        plt.title(f'Scatter plot between {column1} and {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.show()
class Categorical_CategoricalReview(BivariateReview):
    def review(self, data, column1, column2):
        contingency_table = pd.crosstab(data[column1], data[column2])
        print("Contingency Table:")
        print(contingency_table)
        contingency_table.plot(kind='bar', stacked=True, figsize=(10,6))
        plt.title(f'Stacked Bar Chart between {column1} and {column2}')
        plt.xlabel(column1)
        plt.ylabel('Count')
        plt.show()
class Numerical_CategoricalReview(BivariateReview):
    def review(self, data, column1, column2):
        plt.figure(figsize=(10,6))
        sns.boxplot(x=data[column2], y=data[column1])
        plt.title(f'Box plot of {column1} by {column2}')
        plt.xlabel(column2)
        plt.ylabel(column1)
        plt.show()


class SetBivariateStrategy:
    def __init__(self, strategy: BivariateReview):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateReview):
        self._strategy = strategy

    def execute_strategy(self, data: pd.DataFrame, column1: str, column2: str) -> None:
        self._strategy.review(data, column1, column2)