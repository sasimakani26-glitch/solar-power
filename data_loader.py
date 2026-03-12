import pandas as pd

def load_dataset(path):
    return pd.read_csv(path)

def validate_data(df):
    return df.dropna()
