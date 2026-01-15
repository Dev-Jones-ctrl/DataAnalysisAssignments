import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import machine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from machine import FEATURES, prepare_data

def run_improved_machine_learning(df, n_components=5):
    """
    FP-7 Improved Machine Learning:
    - Baseline Logistic Regression (FP-6 reference)
    - PCA-enhanced Logistic Regression (main improved result)
    - Improved Random Forest (secondary improved result, numeric only)
    """
    # Baseline model (FP-6)
    baseline_results = machine.run_machine_learning(df)
    baseline_auc = baseline_results["final_metrics"]["roc_auc"]
    # PCA + Logistic Regression
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
    improved_auc = roc_auc_score(y_test, y_prob)
    # Improved Random Forest (numeric comparison only)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    rf_scores = cross_val_score(
        rf, X, y, cv=5, scoring="roc_auc"
    )
    return {
        "model": model,
        "y_test": y_test,
        "y_prob": y_prob,
        "baseline_auc": baseline_auc,
        "improved_auc": improved_auc,
        "improved_random_forest": {
            "cv_auc_mean": rf_scores.mean(),
            "cv_auc_std": rf_scores.std(),
        }
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

def plot_improved_feature_importance(model, feature_names):
    """
    Fig.6: PCA-Based Feature Importance – Win Probability Model
    """
    pca = model.named_steps["pca"]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    plt.figure(figsize=(7, 5))
    loadings.abs().sum(axis=1).sort_values().plot(kind="barh")
    plt.xlabel("Aggregate Absolute Loading")
    plt.title("Fig.6: PCA-Based Feature Importance")
    plt.tight_layout()
    plt.show()