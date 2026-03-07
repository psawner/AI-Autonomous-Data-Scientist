import pandas as pd

def analyze_data(df):

    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]

    summary["numerical_columns"] = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    summary["categorical_columns"] = df.select_dtypes(include=["object"]).columns.tolist()

    summary["missing_values"] = df.isnull().sum().to_dict()

    return summary