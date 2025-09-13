from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateReview(ABC):
    @abstractmethod
    def review(self, data: pd.DataFrame, column: str = None) -> None:
        pass

class NumericalReview(UnivariateReview):
    def __init__(self, show_all: bool = False):
        self.show_all = show_all

    def _plot_column(self, data: pd.DataFrame, column: str, axes=None) -> None:
        """Helper function to plot a single numerical column on given axes or a new figure."""
        if axes is None:
            # If no axes are provided, create a new figure for a single plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
        # Plot on the provided or newly created axes
        sns.histplot(data[column], kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {column}')
        sns.boxplot(x=data[column], ax=axes[1])
        axes[1].set_title(f'Boxplot of {column}')

        # Only call show() if we created a new figure
        if 'fig' in locals():
            plt.tight_layout()
            plt.show()

    def review(self, data: pd.DataFrame, column: str = None) -> None:
        if self.show_all:
            numerical_cols = data.select_dtypes(include=['number']).columns
            num_cols = len(numerical_cols)

            if num_cols == 0:
                print("No numerical columns to display.")
                return
            num_rows = (num_cols+1) // 2
            fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(12, 5 * num_rows))

            if num_rows == 1:
                axes = [axes] # Make it iterable for the loop
            if num_cols == 1:
                # Hide unused axes if there's an odd number of variables
                axes[0][2].set_visible(False)
                axes[0][3].set_visible(False)
            for i, col in enumerate(numerical_cols):
                row = i // 2
                col_start = (i % 2) * 2
                self._plot_column(data, col, axes=axes[row][col_start:col_start+2])
            if num_cols % 2 != 0 and num_cols > 1:
                axes[-1][2].set_visible(False)
                axes[-1][3].set_visible(False)
            plt.tight_layout()
            plt.show()

        elif column:
            if column in data.select_dtypes(include=['number']).columns:
                # Call without axes to create a new figure
                self._plot_column(data, column)
            else:
                print(f"Error: Column '{column}' is not a numerical column.")
class CategoricalReview(UnivariateReview):
    def review(self, data: pd.DataFrame, column: str) -> None:
        if not column:
            print("Error: Please specify a column for CategoricalReview.")
            return
        plt.figure(figsize=(8, 5))
        sns.countplot(y=data[column], order=data[column].value_counts().index)
        plt.title(f'Count Plot of {column}')
        plt.show()

class SetUnivariateStrategy:
    def __init__(self, strategy: UnivariateReview):
        self._strategy = strategy
    def set_strategy(self, strategy: UnivariateReview) -> None:
        self._strategy = strategy
    def execute_strategy(self, data: pd.DataFrame, column: str = None) -> None:
        self._strategy.review(data, column)