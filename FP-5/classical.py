import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# MAIN RESULT 1
def result_figure_1(df):
    close = df[(df.home_win_prob >= 0.4) & (df.home_win_prob <= 0.6)]
    corr = {}
    for c in df.columns:
        if c.startswith("diff_"):
            x = close[c]
            y = close["home_win_prob"]
            m = x.notna()
            if m.sum() > 10:
                corr[c] = np.corrcoef(x[m], y[m])[0, 1]
    corr = (
        pd.Series(corr)
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(7,4))
    corr.plot(kind="bar")
    plt.ylabel("|Correlation with Home Win Probability|")
    plt.title("Which factors matter most in close games?")
    plt.tight_layout()
    plt.show()

# MAIN RESULT 2
def result_figure_2(df):
    df = df.copy()
    df["total_points"] = df.home_score + df.away_score
    median = df.total_points.median()
    low  = df[df.total_points <= median]["diff_turnover"].dropna()
    high = df[df.total_points > median]["diff_turnover"].dropna()
    t, p = stats.ttest_ind(low, high, equal_var=False)
    plt.figure(figsize=(6,4))
    plt.boxplot([low, high],
                labels=["Low-scoring", "High-scoring"])
    plt.title(f"Turnover impact depends on scoring (p={p:.3f})")
    plt.ylabel("Turnover Differential")
    plt.tight_layout()
    plt.show()