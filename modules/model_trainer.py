from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


import joblib
import os

def detect_problem_type(y):

    if y.dtype == "object" or len(y.unique()) < 20:
        return "classification"
    else:
        return "regression"


def train_models(X, y):

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    best_model = None
    best_score = -999
    best_conf_matrix = None

    if problem_type == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results[name] = acc

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name
                best_conf_matrix = confusion_matrix(y_test, preds)

            os.makedirs("saved_models", exist_ok=True)

            joblib.dump(best_model, "saved_models/best_model.pkl")    

        return problem_type, results, best_model_name, best_model, best_conf_matrix
    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)

            results[name] = r2

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name

            os.makedirs("saved_models", exist_ok=True)

            joblib.dump(best_model, "saved_models/best_model.pkl") 
            
        return problem_type, results, best_model_name, best_model, None