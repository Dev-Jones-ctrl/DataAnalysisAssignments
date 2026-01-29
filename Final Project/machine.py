import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
def evaluate_models(X, y, cv=5):
    """
    Cross-validated comparison of candidate models.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ])
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="roc_auc"
        )
        results[name] = {
            "cv_auc_mean": scores.mean(),
            "cv_auc_std": scores.std(),
        }
    return results
# ML Functions
def run_machine_learning(df):
    """
    Baseline ML analysis with cross-validation.
    """
    X, y = prepare_data(df)
    # 1. Cross-validation (model validation)
    cv_results = evaluate_models(X, y)
    # 2. Train / test evaluation (final result)
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
        "cv_results": cv_results,
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
def plot_final_roc_curve(results):
    """Fig.5: ROC Curve – Win Probability Model"""
    fpr, tpr, _ = roc_curve(results["y_test"], results["y_prob"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Model (AUC = {results['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig.5: ROC Curve – Final Win Probability Model")
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_pca_loadings(improved_model):
    """Fig. 6: """
    pca = improved_model.named_steps["pca"]
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES)
    plt.figure(figsize=(8, 5))
    loadings.abs().sum(axis=1).sort_values().plot(kind="barh", color='skyblue')
    plt.xlabel("Aggregate Absolute Loading")
    plt.title("Fig.6: PCA Structure – Key Performance Factors that Contribute in Combination to Winning")
    plt.show()