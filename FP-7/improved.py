import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from machine import FEATURES, prepare_data

def run_improved_machine_learning(df, n_components=5):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("model", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": model,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_test": y_test,
        "y_prob": y_prob,
    }

def plot_improved_roc_curve(y_true, y_prob):
    """
    Fig.5: ROC Curve – PCA-Enhanced Win Probability Model
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="PCA WP Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig.5: ROC Curve – PCA-Enhanced Win Probability Model")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pca_loadings(model, feature_names):
    """
    Fig.6: PCA Loadings – Key Performance Dimensions
    (visual only, no table output)
    """
    pca = model.named_steps["pca"]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    plt.figure(figsize=(8, 5))
    loadings.abs().sum(axis=1).sort_values().plot(kind="barh")
    plt.xlabel("Aggregate Absolute Loading")
    plt.title("Fig.6: PCA Loadings – Key Performance Dimensions")
    plt.tight_layout()
    plt.show()

def get_pca_loadings(model, feature_names):
    """
    PCA loading table for analysis details only (not main report)
    """
    pca = model.named_steps["pca"]
    return pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )