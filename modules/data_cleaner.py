import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def clean_data(df, target_column):

    df = df.copy()

    # Separate target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Handle missing values
    for col in X.columns:
        if X[col].dtype == "object":
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)

    # Encode categorical columns
    label_encoders = {}
    
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y