import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df, target_column):

    df = df.copy()
    warnings = []
    # target missing value check
    target_missing_ratio = df[target_column].isnull().mean()

    if target_missing_ratio > 0.10:
        warnings.append(
            f"Target column '{target_column}' has {target_missing_ratio*100:.2f}% missing values. "
            "Model training may be unreliable. Consider collecting more data."
        )

    # If <=10%, remove those rows
    df = df.dropna(subset=[target_column])

    y = df[target_column]
    X = df.drop(columns=[target_column])

    feature_missing_ratio = X.isnull().mean()

    highly_missing_cols = feature_missing_ratio[feature_missing_ratio > 0.80]

    if not highly_missing_cols.empty:
        warnings.append(
            f"The following columns have more than 80% missing values: {list(highly_missing_cols.index)}. "
            "Consider collecting better data."
        )

    # handling missing values
    for col in X.columns:

        if X[col].dtype == "object":

            mode_val = X[col].mode()

            if not mode_val.empty:
                X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna("Unknown")

        else:
            X[col] = X[col].fillna(X[col].mean())

    # encoding 
    label_encoders = {}

    for col in X.select_dtypes(include="object").columns:

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # encoding target column
    if y.dtype == "object" or y.dtype == "category":

        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y, warnings




