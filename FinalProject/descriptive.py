import matplotlib.pyplot as plt
import numpy as np

# Fig1:SCORE DIFFERENTIAL DISTRIBUTION
def figure_1_score_distribution(df):
    """
    Fig.1:
    Distribution of score differentials (Home − Away).
    Provides a global overview of game outcomes.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(df["score_diff"], bins=30)
    plt.xlabel("Score Differential (Home − Away)")
    plt.ylabel("Number of Games")
    plt.title("Fig.1: Distribution of Game Score Differentials")
    plt.tight_layout()
    plt.show()
# Fig2:KEY EFFICIENCY VS SCORE DIFFERENTIAL
def figure_2_key_efficiency_relationship(df):
    """
    Fig.2:
    Relationship between Passing EPA Differential and Score Differential.
    Illustrates how offensive efficiency translates into points.
    """
    x = df["diff_pass_epa"]
    y = df["score_diff"]
    mask = x.notna() & y.notna()
    plt.figure(figsize=(6, 4))
    plt.scatter(x[mask], y[mask], alpha=0.4)
    # Linear trend (visual aid only)
    m, b = np.polyfit(x[mask], y[mask], 1)
    plt.plot(x[mask], m * x[mask] + b)
    plt.xlabel("Passing EPA Differential (Home − Away)")
    plt.ylabel("Score Differential (Home − Away)")
    plt.title("Fig.2: Passing Efficiency and Game Outcomes")
    plt.tight_layout()
    plt.show()