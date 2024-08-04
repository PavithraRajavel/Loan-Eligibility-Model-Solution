import pandas as pd


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise IOError(f"Error loading data from {file_path}: {e}")
