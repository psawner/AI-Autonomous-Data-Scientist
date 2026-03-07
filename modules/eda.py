import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


sns.set_style("whitegrid")

def plot_target_distribution(y, save=False):

    fig, ax = plt.subplots(figsize=(5,3))

    if y.dtype == "object" or y.nunique() < 15:
        sns.countplot(x=y, ax=ax)
    else:
        sns.histplot(y, kde=True, ax=ax)

    ax.set_title("Target Distribution")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save:
        os.makedirs("saved_charts", exist_ok=True)
        fig.savefig("saved_charts/target_distribution.png")

    return fig



def plot_correlation_heatmap(X, save=False):

    numeric = X.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(10,6))

    corr = numeric.corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=False,
        linewidths=0.8,
        ax=ax
    )

    ax.set_title("Feature Correlation")

    plt.tight_layout()

    if save:
        os.makedirs("saved_charts", exist_ok=True)
        fig.savefig("saved_charts/correlation_heatmap.png")

    return fig



def plot_feature_distribution(X, save=False):

    figs = []

    numeric = X.select_dtypes(include="number")
    categorical = X.select_dtypes(exclude="number")

    # Numeric distributions
    for col in numeric.columns[:3]:

        fig, ax = plt.subplots(figsize=(5,3))

        sns.histplot(X[col], kde=True, ax=ax)

        ax.set_title(f"{col} Distribution")

        plt.tight_layout()

        if save:
            os.makedirs("saved_charts", exist_ok=True)
            fig.savefig(f"saved_charts/feature_dist_{col}.png")

        figs.append(fig)

    # Categorical distributions
    for col in categorical.columns[:2]:

        fig, ax = plt.subplots(figsize=(5,3))

        sns.countplot(x=X[col], ax=ax)

        ax.set_title(f"{col} Counts")

        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            os.makedirs("saved_charts", exist_ok=True)
            fig.savefig(f"saved_charts/feature_count_{col}.png")

        figs.append(fig)

    return figs