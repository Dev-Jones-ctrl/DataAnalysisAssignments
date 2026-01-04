import pandas as pd
import numpy as np

def load_data(filepath):
    dfs = []
    for year in range(2017, 2026):
        dfs.append(pd.read_csv(f"{filepath}_{year}.csv", low_memory=False))
    return pd.concat(dfs, ignore_index=True)
    
# GAME-LEVEL OUTCOMES (DEPENDENT VARIABLE)
def make_game_outcomes(df):
    games = (
        df.groupby("game_id")
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_score=("home_score", "max"),
            away_score=("away_score", "max"),
        )
        .reset_index()
    )
    games["score_diff"] = games["home_score"] - games["away_score"]
    games["home_win"] = (games["score_diff"] > 0).astype(int)
    return games

# INDEPENDENT VARIABLES (GAME-LEVEL)
def passing_epa(df):
    return (
        df[df["pass_attempt"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .mean()
        .unstack()
    )

def rushing_epa(df):
    return (
        df[df["rush_attempt"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .mean()
        .unstack()
    )

def redzone_score_rate(df):
    rz = df[df["yardline_100"] <= 20]
    drives = (
        rz.groupby(["game_id", "posteam", "drive"])["fixed_drive_result"]
        .first()
    )
    scored = drives.isin(["Touchdown", "Field Goal"]).astype(int)
    return (
        scored.groupby(["game_id", "posteam"])
        .mean()
        .unstack()
    )

def sack_rate(df):
    df = df.copy()
    df["dropback"] = df["pass_attempt"] + df["sack"]
    return (
        df.groupby(["game_id", "defteam"])
        .apply(lambda g: g["sack"].sum() / max(1, g["dropback"].sum()))
        .unstack()
    )

def defensive_pass_epa(df):
    return (
        -df[df["pass_attempt"] == 1]
        .groupby(["game_id", "defteam"])["epa"]
        .mean()
        .unstack()
    )

def turnover_margin(df):
    giveaways = (
        df.groupby(["game_id", "posteam"])
        .apply(lambda g: g["interception"].sum() + g["fumble_lost"].sum())
    )
    takeaways = (
        df.groupby(["game_id", "defteam"])
        .apply(lambda g: g["interception"].sum() + g["fumble_lost"].sum())
    )
    return (takeaways - giveaways).unstack()

def third_down_conversion(df):
    td = df[df["down"] == 3]
    return (
        td.groupby(["game_id", "posteam"])["third_down_converted"]
        .mean()
        .unstack()
    )

def time_of_possession(df):
    drives = (
        df.groupby(["game_id", "posteam", "drive"])["drive_time_of_possession"]
        .first()
        .dropna()
    )
    def to_seconds(x):
        m, s = map(int, x.split(":"))
        return 60 * m + s
    return (
        drives.apply(to_seconds)
        .groupby(["game_id", "posteam"])
        .sum()
        .unstack()
    )

def penalty_differential(df):
    committed = (
        df.groupby(["game_id", "posteam"])["penalty_yards"]
        .sum()
    )
    received = (
        df.groupby(["game_id", "defteam"])["penalty_yards"]
        .sum()
    )
    return (committed - received).unstack()

def field_goal_epa(df):
    return (
        df[df["field_goal_attempt"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .mean()
        .unstack()
    )

# FINAL DATASET BUILDER
def build_full_dataset(filepath):
    df = load_data(filepath)
    games = make_game_outcomes(df)
    iv_tables = {
        "pass_epa": passing_epa(df),
        "rush_epa": rushing_epa(df),
        "rz": redzone_score_rate(df),
        "sack": sack_rate(df),
        "def_pass_epa": defensive_pass_epa(df),
        "turnover": turnover_margin(df),
        "third_down": third_down_conversion(df),
        "top": time_of_possession(df),
        "penalty": penalty_differential(df),
        "st_epa": field_goal_epa(df),
    }
    for name, table in iv_tables.items():
        games[f"diff_{name}"] = games.apply(
            lambda r: (
                table.at[r["game_id"], r["home_team"]]
                - table.at[r["game_id"], r["away_team"]]
                if r["game_id"] in table.index
                and r["home_team"] in table.columns
                and r["away_team"] in table.columns
                else np.nan
            ),
            axis=1,
        )
    return games