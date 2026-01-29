import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# Data Prep
FEATURES = [
    "diff_pass_epa", "diff_rush_epa", "diff_rz", "diff_sack",
    "diff_def_pass_epa", "diff_turnover", "diff_third_down",
    "diff_top", "diff_penalty", "diff_st_epa",
]
def prepare_data(df):
    X = df[FEATURES]
    y = df["home_win"]
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    return X.loc[mask], y.loc[mask]
# ML Functions
def run_machine_learning(df):
    """Originale FP-6 Analyse (Baseline)"""
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    ) 
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": model,
        "y_test": y_test,
        "y_prob": y_prob,
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
def run_improved_machine_learning(df, n_components=5):
    """Verbesserte FP-7 Analyse (PCA)"""
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
        "y_test": y_test,
        "y_prob": y_prob,
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
# Plotting
def plot_roc_curves(baseline_results, improved_results):
    """Fig. 5: Vergleich der ROC-Kurven"""
    plt.figure(figsize=(6, 5))  
    # Baseline
    fpr_b, tpr_b, _ = roc_curve(baseline_results["y_test"], baseline_results["y_prob"])
    plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC = {baseline_results['roc_auc']:.3f})")   
    # Improved
    fpr_i, tpr_i, _ = roc_curve(improved_results["y_test"], improved_results["y_prob"])
    plt.plot(fpr_i, tpr_i, label=f"PCA Improved (AUC = {improved_results['roc_auc']:.3f})")   
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig.5: ROC Curve Comparison")
    plt.legend()
    plt.show()
def plot_pca_loadings(improved_model):
    """Fig. 6: PCA Feature Importance (aus improved.py)"""
    pca = improved_model.named_steps["pca"]
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES)
    plt.figure(figsize=(8, 5))
    loadings.abs().sum(axis=1).sort_values().plot(kind="barh", color='skyblue')
    plt.xlabel("Aggregate Absolute Loading")
    plt.title("Fig.6: PCA Feature Importance")
    plt.show()
def plot_logistic_feature_importance(baseline_model):
    """Fig. 7: Standard Feature Importance (aus machine.py)"""
    coef = baseline_model.named_steps["model"].coef_[0]
    importance = pd.Series(coef, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color='salmon')
    plt.xlabel("Standardized Coefficient")
    plt.title("Fig.7: Feature Importance (Baseline Model)")
    plt.show()