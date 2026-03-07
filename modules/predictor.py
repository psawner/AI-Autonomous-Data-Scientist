import joblib
import pandas as pd


def load_model():
    model = joblib.load("saved_models/best_model.pkl")
    return model


def make_prediction(input_data):

    model = load_model()

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)

    return prediction[0]