import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def median_split(series):
    med = series.median()
    return series > med
def quartile_split(series, q=0.25):
    low = series <= series.quantile(q)
    high = series >= series.quantile(1 - q)
    return low, high
def ttest_groups(x, y):
    return stats.ttest_ind(x, y, nan_policy="omit", equal_var=False)

# H1:TURNOVER MARGIN
def turnover_margin_tests(df):
    """
    Explore different ways of testing turnover margin.
    """
    results = {}
    # Zero split: positive vs non-positive
    pos = df[df["diff_turnover"] > 0]["score_diff"]
    neg = df[df["diff_turnover"] <= 0]["score_diff"]
    results["zero_split"] = ttest_groups(pos, neg)
    # Quartile split
    low, high = quartile_split(df["diff_turnover"])
    results["quartile_split"] = ttest_groups(
        df.loc[high, "score_diff"],
        df.loc[low, "score_diff"]
    )
    return results

def plot_turnover_margin(df):
    """
    FIGURE CANDIDATE:
    Score Differential by Turnover Margin (Positive vs Negative)
    """
    df_plot = df.copy()
    df_plot["Turnover Margin"] = np.where(
        df_plot["diff_turnover"] > 0,
        "Positive",
        "Zero / Negative"
    )
    plt.figure(figsize=(6, 4))
    df_plot.boxplot(
        column="score_diff",
        by="Turnover Margin",
        grid=False
    )
    plt.title("Fig.3: Score Differential by Turnover Margin")
    plt.suptitle("")
    plt.xlabel("")
    plt.ylabel("Score Differential (Home − Away)")
    plt.tight_layout()
    plt.show()

# H2: RED ZONE EFFICIENCY

def redzone_tests(df):
    """
    Test different splits of Red Zone differential.
    """
    results = {}
    # Median split
    high = df[median_split(df["diff_rz"])]["score_diff"]
    low = df[~median_split(df["diff_rz"])]["score_diff"]
    results["median_split"] = ttest_groups(high, low)

    # Quartile split (more insightful)
    low_q, high_q = quartile_split(df["diff_rz"])
    results["quartile_split"] = ttest_groups(
        df.loc[high_q, "score_diff"],
        df.loc[low_q, "score_diff"]
    )
    return results
def plot_redzone(df):
    """
    FIGURE CANDIDATE:
    Score Differential by Red Zone Efficiency (Top vs Bottom Quartile)
    """
    low, high = quartile_split(df["diff_rz"])
    df_plot = df.loc[low | high].copy()
    df_plot["Red Zone Efficiency"] = np.where(
        high[low | high],
        "Top 25%",
        "Bottom 25%"
    )
    plt.figure(figsize=(6, 4))
    df_plot.boxplot(
        column="score_diff",
        by="Red Zone Efficiency",
        grid=False
    )
    plt.title("Fig.4: Score Differential by Red Zone Efficiency")
    plt.suptitle("")
    plt.xlabel("")
    plt.ylabel("Score Differential (Home − Away)")
    plt.tight_layout()
    plt.show()

# H3: SPECIAL TEAMS EPA (EXPLORATORY)
def special_teams_tests(df):
    """
    Exploratory tests for Special Teams EPA.
    """
    results = {}
    # Median split
    high = df[median_split(df["diff_st_epa"])]["score_diff"]
    low = df[~median_split(df["diff_st_epa"])]["score_diff"]
    results["median_split"] = ttest_groups(high, low)
    # Correlation
    results["correlation"] = stats.pearsonr(
        df["diff_st_epa"].dropna(),
        df.loc[df["diff_st_epa"].notna(), "score_diff"]
    )
    return results

# H4: COMBINED GAME CONTROL METRICS
def game_control_tests(df):
    """
    Combined exploratory analysis of TOP and Penalties.
    """
    results = {}
    control_index = (
        df["diff_top"].rank(pct=True)
        - df["diff_penalty"].rank(pct=True)
    )
    high = df[control_index > control_index.median()]["score_diff"]
    low = df[control_index <= control_index.median()]["score_diff"]
    results["composite_control"] = ttest_groups(high, low)
    return results

def run_all_hypothesis_tests(df):
    """
    Run all hypothesis tests.
    Only a subset should be reported in the main notebook.
    """
    results = {
        "turnover": turnover_margin_tests(df),
        "redzone": redzone_tests(df),
        "special_teams": special_teams_tests(df),
        "game_control": game_control_tests(df),
    }
    return results