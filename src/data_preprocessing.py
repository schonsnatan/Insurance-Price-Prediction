import pandas as pd
import numpy as np

def load_data(filepath):
    """Carrega os dados a partir do arquivo CSV"""
    df = pd.read_csv(filepath, header=None)
    headers = ['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']
    df.columns = headers
    return df

def clean_data(df):
    """Trata valores ausentes e ajusta tipos de dados"""
    df.replace('?', np.nan, inplace=True)
    df['age'] = df['age'].fillna(df['age'].astype(float).mean())
    df['smoker'] = df['smoker'].fillna(df['smoker'].value_counts().idxmax())
    df[['age', 'smoker']] = df[['age', 'smoker']].astype(int)
    df['charges'] = df['charges'].round(2)
    return df
