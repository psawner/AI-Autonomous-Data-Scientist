from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    r2_score
)


def detect_problem_type(y):

    if y.dtype == "object" or len(y.unique()) < 20:
        return "classification"
    else:
        return "regression"
    
def is_scale(X_train, X_test):

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test


def is_imbalanced(y):

    class_ratio = y.value_counts(normalize=True)

    if class_ratio.min() < 0.35:
        return True
    return False


def handle_outliers(X):

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X = X.clip(lower=lower, upper=upper, axis=1)

    return X


def cross_validation_score(model, X_train, y_train, problem_type):

    if problem_type == "classification":
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    return scores.mean(), scores.std()


def evaluate_classification(y_test, preds, probs=None, imbalanced=False):

    metrics = {}

    if not imbalanced:
        metrics["accuracy"] = accuracy_score(y_test, preds)

    metrics["precision"] = precision_score(y_test, preds, average="weighted")
    metrics["recall"] = recall_score(y_test, preds, average="weighted")
    metrics["f1_score"] = f1_score(y_test, preds, average="weighted")

    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, probs, multi_class="ovr")
        except:
            pass

    return metrics


def train_models(X, y):

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = handle_outliers(X_train)
    X_test = handle_outliers(X_test)

    results = {}
    best_model = None
    best_model_name = None
    best_score = -999
    best_conf_matrix = None

    if problem_type == "classification":

        imbalanced = is_imbalanced(y_train)

        # Handle class imbalance
        if imbalanced:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        for name, model in models.items():

            if name == "Logistic Regression":
                X_train, X_test = is_scale(X_train, X_test)

            model.fit(X_train, y_train)

            # cross validation
            cv_mean, cv_std = cross_validation_score(model, X_train, y_train, problem_type)

            preds = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
            else:
                probs = None

            metrics = evaluate_classification(y_test, preds, probs, imbalanced)

            results[name] = {
                "metrics": metrics,
                "cv_mean": cv_mean,
                "cv_std": cv_std
            }

            score = cv_mean

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
                best_conf_matrix = confusion_matrix(y_test, preds)

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }

        for name, model in models.items():

            if name == "Linear Regression":
                X_train, X_test = is_scale(X_train, X_test)

            model.fit(X_train, y_train)

            # cross validation
            cv_mean, cv_std = cross_validation_score(model, X_train, y_train, problem_type)

            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)

            results[name] = {
                "metrics": r2,
                "cv_mean": cv_mean,
                "cv_std": cv_std
            }
            
            score = cv_mean

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, "saved_models/best_model.pkl")

    return problem_type, results, best_model_name, best_model, best_conf_matrix