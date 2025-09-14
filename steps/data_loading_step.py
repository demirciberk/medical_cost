from src.data_loader import DataLoader
from zenml import step
import pandas as pd

@step
def data_loader() -> pd.DataFrame:
    loader = DataLoader()
    return loader.load_data()

    