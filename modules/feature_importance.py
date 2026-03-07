import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_feature_importance(model, X):

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        })

        feature_importance = feature_importance.sort_values(
            by="importance",
            ascending=False
        )

        return feature_importance

    else:
        return None


def plot_feature_importance(feature_importance):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance.head(10),
        ax=ax
    )

    ax.set_title("Top Feature Importances")

    return fig