import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
)
# DATA PREP
FEATURES = [
    "diff_pass_epa",
    "diff_rush_epa",
    "diff_rz",
    "diff_sack",
    "diff_def_pass_epa",
    "diff_turnover",
    "diff_third_down",
    "diff_top",
    "diff_penalty",
    "diff_st_epa",
]

def prepare_data(df):
    X = df[FEATURES]
    y = df["home_win"]
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    return X.loc[mask], y.loc[mask]

# MODEL EXPLORATION
def evaluate_models(X, y):
    """
    Explore several ML models with cross-validation.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=5, scoring="roc_auc"
        )
        results[name] = {
            "cv_auc_mean": scores.mean(),
            "cv_auc_std": scores.std(),
        }
    return results
# FINAL MODEL: LOGISTIC REGRESSION WP MODEL
def train_final_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    return model, X_test, y_test, y_prob, metrics
# FIG ROC CURVE MAIN RESULT
def plot_roc_curve(y_true, y_prob):
    """
    FIGURE CANDIDATE (ONLY ONE FOR MAIN NOTEBOOK):
    ROC Curve of Win Probability Model
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="WP Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig.5: ROC Curve – Win Probability Model")
    plt.legend()
    plt.tight_layout()
    plt.show()

# FEATURE IMPORTANCE (INTERPRETATION)
def logistic_feature_importance(model, feature_names):
    """
    Return standardized coefficients for interpretation.
    """
    coef = model.named_steps["model"].coef_[0]
    return pd.Series(coef, index=feature_names).sort_values()
    
# MASTER FUNCTION
def run_machine_learning(df):
    """
    Full ML workflow.
    Only the ROC curve should be reported in the main notebook.
    """
    X, y = prepare_data(df)
    exploration = evaluate_models(X, y)
    model, X_test, y_test, y_prob, metrics = train_final_model(X, y)
    feature_importance = logistic_feature_importance(model, FEATURES)
    return {
        "model_comparison": exploration,
        "final_metrics": metrics,
        "feature_importance": feature_importance,
        "y_test": y_test,
        "y_prob": y_prob,
    }