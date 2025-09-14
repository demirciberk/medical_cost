import logging
import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class DataLoader():
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv('../data/insurance.csv')
        