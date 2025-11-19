#parse data module
import pandas as pd
import numpy as np

def load_data(filepath):
    df_plays_2017 = pd.read_csv(filepath+'_2017.csv', low_memory=False)
    df_plays_2018 = pd.read_csv(filepath+'_2018.csv', low_memory=False)
    df_plays_2019 = pd.read_csv(filepath+'_2019.csv', low_memory=False)
    df_plays_2020 = pd.read_csv(filepath+'_2020.csv', low_memory=False)
    df_plays_2021 = pd.read_csv(filepath+'_2021.csv', low_memory=False)
    df_plays_2022 = pd.read_csv(filepath+'_2022.csv', low_memory=False)
    df_plays_2023 = pd.read_csv(filepath+'_2023.csv', low_memory=False)
    df_plays_2024 = pd.read_csv(filepath+'_2024.csv', low_memory=False)
    df_plays_2025 = pd.read_csv(filepath+'_2025.csv', low_memory=False)
    # combine all the data verticaly with pd.concat. And use ignore_index so that the index from play 1 in 2017 isnt the same as from play 1 in 2025 etc.
    df_plays = pd.concat([
        df_plays_2017, df_plays_2018, df_plays_2019, df_plays_2020, 
        df_plays_2021, df_plays_2022, df_plays_2023, df_plays_2024, 
        df_plays_2025
    ], ignore_index=True)
    return df_plays

    
def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)

def compute_diff(row, df_iv):
    gid = row['game_id']
    home, away = row['home_team'], row['away_team']

    if gid not in df_iv.index:
        return np.nan
    if home not in df_iv.columns or away not in df_iv.columns:
        return np.nan

    try:
        h = float(df_iv.at[gid, home])
        a = float(df_iv.at[gid, away])
        return h - a
    except:
        return np.nan


def make_game_results(df_plays):
    game_results = df_plays.groupby('game_id').agg(
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
        home_score=('home_score', 'max'),
        away_score=('away_score', 'max')
    ).reset_index()

    game_results['home_win_prob'] = game_results.apply(
        lambda r: r['home_score'] / (r['home_score'] + r['away_score'])
        if (r['home_score'] + r['away_score']) > 0 else np.nan,
        axis=1
    )
    return game_results


def passing_offense(df_plays):
    return (
        df_plays[df_plays['pass_attempt'] == 1]
        .groupby(['game_id', 'posteam'])['epa']
        .mean()
        .unstack()
    )


def rushing_offense(df_plays):
    return (
        df_plays[df_plays['rush_attempt'] == 1]
        .groupby(['game_id', 'posteam'])['epa']
        .mean()
        .unstack()
    )


def red_zone_offense(df_plays):
    rz = df_plays[df_plays['yardline_100'] <= 20]
    rz_scores = (
        rz.groupby(['game_id', 'posteam'])
        .apply(lambda g: (g['touchdown'].sum() + g['field_goal_result'].isin(['made']).sum()) / len(g))
        .reset_index(name='rz_score')
        .pivot(index='game_id', columns='posteam', values='rz_score')
    )
    return rz_scores


def defensive_pressure(df_plays):
    df = (
        df_plays.groupby(['game_id', 'defteam'])
        .apply(lambda g: g['sack'].sum() / max(1, g['pass_attempt'].sum()))
        .reset_index(name='sack_rate')
        .pivot(index='game_id', columns='defteam', values='sack_rate')
    )
    return df


def defensive_pass_epa(df_plays):
    return (
        df_plays[df_plays['pass_attempt'] == 1]
        .groupby(['game_id', 'defteam'])['epa']
        .mean()
        .unstack()
    )


def turnovers(df_plays):
    return (
        df_plays.groupby(['game_id', 'posteam'])
        .apply(lambda g: (g['interception'].sum() + g['fumble_lost'].sum()) * -1)
        .reset_index(name='turnover_margin')
        .pivot(index='game_id', columns='posteam', values='turnover_margin')
    )


def third_down(df_plays):
    cond = (df_plays['third_down_converted'] == 1) | (df_plays['third_down_failed'] == 1)
    return (
        df_plays[cond]
        .groupby(['game_id', 'posteam'])
        .apply(lambda g: g['third_down_converted'].sum() / len(g))
        .reset_index(name='third_down_conv')
        .pivot(index='game_id', columns='posteam', values='third_down_conv')
    )


def field_goal_epa(df_plays):
    return (
        df_plays[df_plays['field_goal_attempt'] == 1]
        .groupby(['game_id', 'posteam'])['epa']
        .mean()
        .unstack()
    )


def time_of_possession(df_plays):
    if 'drive_time_of_possession' in df_plays.columns:
        return (
            df_plays.groupby(['game_id', 'posteam'])['drive_time_of_possession']
            .sum()
            .unstack()
        )
    else:
        return (
            df_plays.groupby(['game_id', 'posteam'])['drive']
            .count()
            .unstack()
        )


def penalty_yards(df_plays):
    return (
        df_plays.groupby(['game_id', 'posteam'])['penalty_yards']
        .sum()
        .unstack()
    )


def build_game_iv_dataset(df_plays):
    game_results = make_game_results(df_plays)

    # Dictionary of all metric functions
    iv_funcs = {
        'pass_off': passing_offense,
        'rush_off': rushing_offense,
        'rz': red_zone_offense,
        'sack': defensive_pressure,
        'def_pass': defensive_pass_epa,
        'turnover': turnovers,
        'third_down': third_down,
        'fg': field_goal_epa,
        'top': time_of_possession,
        'pen': penalty_yards
    }

    # Compute metric tables
    iv_tables = {name: fn(df_plays).apply(pd.to_numeric, errors='coerce')
                 for name, fn in iv_funcs.items()}

    # Apply home-away differentials
    for name, table in iv_tables.items():
        game_results[f"diff_{name}"] = game_results.apply(
            lambda r: compute_diff(r, table), axis=1
        )

    return game_results, iv_tables


def build_full_dataset(filepath):
    df_plays = load_data(filepath)
    df_final, _ = build_game_iv_dataset(df_plays)

    return df_final
