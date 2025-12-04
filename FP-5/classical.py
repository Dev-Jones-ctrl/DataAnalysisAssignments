import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from parse_data import build_full_dataset

df = build_full_dataset("data/nfl")   # adjust path if needed
df = df.dropna(subset=["home_win_prob"])
df["total_points"] = df["home_score"] + df["away_score"]
# ANALYSIS 1: Close games only
def close_game_correlations(df, low=0.4, high=0.6):
    close = df[(df.home_win_prob >= low) & (df.home_win_prob <= high)]
    corr = {}
    for c in df.columns:
        if c.startswith("diff_"):
            x = close[c]
            y = close["home_win_prob"]
            mask = x.notna()
            if mask.sum() > 10:
                corr[c] = np.corrcoef(x[mask], y[mask])[0, 1]
    return (
        pd.Series(corr)
        .dropna()
        .sort_values(key=np.abs, ascending=False)
    )
corr_close = close_game_correlations(df)
# ANALYSIS 2: Turnovers depend on scoring environment
median_points = df["total_points"].median()
low_scoring  = df[df.total_points <= median_points]
high_scoring = df[df.total_points > median_points]
turn_low  = low_scoring["diff_turnover"].dropna()
turn_high = high_scoring["diff_turnover"].dropna()
tstat, pval = stats.ttest_ind(turn_low, turn_high, equal_var=False)
print("Turnover differential t-test:")
print(f"t = {tstat:.3f}, p = {pval:.4f}")
