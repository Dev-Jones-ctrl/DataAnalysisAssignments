from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd


def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )

def central(x, print_output=True):
    x0 = np.mean(x)
    x1 = np.median(x)
    x2 = stats.mode(x).mode
    return x0, x1, x2


def dispersion(x, print_output=True):
    y0 = np.std(x)
    y1 = np.min(x)
    y2 = np.max(x)
    y3 = y2 - y1
    y4 = np.percentile(x, 25)
    y5 = np.percentile(x, 75)
    y6 = y5 - y4
    return y0, y1, y2, y3, y4, y5, y6


def display_central_tendency_table(df, num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = df.select_dtypes(include=[np.number]).apply(lambda x: central(x), axis=0)
    round_dict = {
        'home_score': 1,
        'away_score': 1,
        'home_win_prob': 4,
        'diff_pass_off': 6,
        'diff_rush_off': 6,
        'diff_rz': 6,
        'diff_sack': 6,
        'diff_def_pass': 6,
        'diff_turnover': 6,
        'diff_third_down': 6,
        'diff_fg': 6,
        'diff_top': 3,
        'diff_pen': 6
    }
    df_central = df_central.round(round_dict)
    row_labels = ('mean', 'median', 'mode')
    df_central.index = row_labels
    display(df_central)


def display_dispersion_table(df, num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    numeric_df = df.select_dtypes(include=[np.number])
    round_dict = {
        'home_score': 1,
        'away_score': 1,
        'home_win_prob': 4,
        'diff_pass_off': 3,
        'diff_rush_off': 3,
        'diff_rz': 3,
        'diff_sack': 3,
        'diff_def_pass': 3,
        'diff_turnover': 3,
        'diff_third_down': 3,
        'diff_fg': 3,
        'diff_top': 3,
        'diff_pen': 3
    }
    df_dispersion = numeric_df.apply(lambda x: dispersion(x), axis=0).round(round_dict)
    row_labels_dispersion = ['st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR']
    df_dispersion.index = row_labels_dispersion
    display(df_dispersion)


def extract_variables(df):
    home_score = df['home_score']
    away_score = df['away_score']
    y = df['home_win_prob']
    diff_pass_off = df['diff_pass_off']
    diff_rush_off = df['diff_rush_off']
    diff_rz = df['diff_rz']
    diff_sack = df['diff_sack']
    diff_def_pass = df['diff_def_pass']
    diff_turnover = df['diff_turnover']
    diff_third_down = df['diff_third_down']
    diff_fg = df['diff_fg']
    diff_top = df['diff_top']
    diff_pen = df['diff_pen']
    return home_score, away_score, y, diff_pass_off, diff_rush_off, diff_rz, diff_sack, diff_def_pass, diff_turnover, diff_third_down, diff_fg, diff_top, diff_pen


def plot_scatter_grid(df):
    fig, axs = plt.subplots(2, 5, figsize=(18, 7), tight_layout=True)
    axs[0, 0].scatter(df['diff_pass_off'], df['home_win_prob'], alpha=0.5, color='b')
    axs[0, 1].scatter(df['diff_rush_off'], df['home_win_prob'], alpha=0.5, color='r')
    axs[0, 2].scatter(df['diff_rz'], df['home_win_prob'], alpha=0.5, color='g')
    axs[0, 3].scatter(df['diff_sack'], df['home_win_prob'], alpha=0.5, color='c')
    axs[0, 4].scatter(df['diff_def_pass'], df['home_win_prob'], alpha=0.5, color='m')
    axs[1, 0].scatter(df['diff_turnover'], df['home_win_prob'], alpha=0.5, color='y')
    axs[1, 1].scatter(df['diff_third_down'], df['home_win_prob'], alpha=0.5, color='k')
    axs[1, 2].scatter(df['diff_fg'], df['home_win_prob'], alpha=0.5, color='orange')
    axs[1, 3].scatter(df['diff_top'], df['home_win_prob'], alpha=0.5, color='purple')
    axs[1, 4].scatter(df['diff_pen'], df['home_win_prob'], alpha=0.5, color='brown')
    xlabels = (
        'Pass Off EPA', 'Rush Off EPA', 'Red Zone %', 'Sack Rate', 'Def Pass EPA',
        'Turnover Margin', '3rd Down %', 'FG EPA', 'Time of Poss.', 'Penalty Yds'
    )
    axs_list = axs.flatten()
    [ax.set_xlabel(lbl) for ax, lbl in zip(axs_list, xlabels)]
    axs[0, 0].set_ylabel('Home Win Probability')
    axs[1, 0].set_ylabel('Home Win Probability')
    [ax.set_yticklabels([]) for ax in axs_list[1:] if ax not in [axs[0, 0], axs[1, 0]]]
    plt.show()


def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0, 1]
    return r


def plot_regression_line(ax, x, y, **kwargs):
    a, b = np.polyfit(x, y, deg=1)
    x0, x1 = min(x), max(x)
    y0, y1 = a * x0 + b, a * x1 + b
    ax.plot([x0, x1], [y0, y1], **kwargs)


def plot_scatter_with_regression(df):
    fig, axs = plt.subplots(2, 5, figsize=(18, 7), tight_layout=True)
    ivs = [
        df['diff_pass_off'], df['diff_rush_off'], df['diff_rz'], df['diff_sack'], df['diff_def_pass'],
        df['diff_turnover'], df['diff_third_down'], df['diff_fg'], df['diff_top'], df['diff_pen']
    ]
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    axs_list = axs.flatten()
    for ax, x, c in zip(axs_list, ivs, colors):
        ax.scatter(x, df['home_win_prob'], alpha=0.5, color=c)
        x_clean = x.replace([np.inf, -np.inf], np.nan).dropna()
        y_clean = df['home_win_prob'][x_clean.index]
        if len(x_clean) > 1 and x_clean.std() > 0:
            plot_regression_line(ax, x_clean, y_clean, color='k', ls='-', lw=2)
            r = corrcoeff(x_clean, y_clean)
            ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
    xlabels = (
        'Pass Off EPA', 'Rush Off EPA', 'Red Zone %', 'Sack Rate', 'Def Pass EPA',
        'Turnover Margin', '3rd Down %', 'FG EPA', 'Time of Poss.', 'Penalty Yds'
    )
    [ax.set_xlabel(s) for ax, s in zip(axs_list, xlabels)]
    axs[0, 0].set_ylabel('Home Win Probability')
    axs[1, 0].set_ylabel('Home Win Probability')
    [ax.set_yticklabels([]) for ax in axs_list[1:] if ax not in [axs[0, 0], axs[1, 0]]]
    plt.show()


def plot_penalty_split(df):
    i_low = df['home_win_prob'] <= 0.5
    i_high = df['home_win_prob'] > 0.5
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
    i = [df['diff_pen']]
    for ax, i in zip(axs, [i_low, i_high]):
        ax.scatter(df['diff_pen'][i], df['home_win_prob'][i], alpha=0.5, color='g')
        plot_regression_line(ax, df['diff_pen'][i], df['home_win_prob'][i], color='k', ls='-', lw=2)
    [ax.set_xlabel('Penalty Yards') for ax in axs]
    axs[0].set_title('Low win probability')
    axs[0].set_ylabel('Home Win Probability')
    axs[1].set_title('High win probability')
    plt.show()


def plot_penalty_split_with_means(df):
    i_low = df['home_win_prob'] <= 0.5
    i_high = df['home_win_prob'] > 0.5
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
    for ax, i in zip(axs, [i_low, i_high]):
        ax.scatter(df['diff_pen'][i], df['home_win_prob'][i], alpha=0.5, color='g')
        plot_regression_line(ax, df['diff_pen'][i], df['home_win_prob'][i], color='k', ls='-', lw=2)
    [axs[0].plot(df['diff_pen'][i_low].mean(), p, 'ro') for p in [0.1, 0.2, 0.3]]
    [axs[1].plot(df['diff_pen'][i_high].mean(), p, 'ro') for p in [0.6, 0.7, 0.8]]
    [ax.set_xlabel('Penalty Yards') for ax in axs]
    axs[0].set_title('Low win probability')
    axs[0].set_ylabel('Home Win Probability')
    axs[1].set_title('High win probability')
    plt.show()


def plot_descriptive_final(df):
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), tight_layout=True)
    ivs = [
        df['diff_pass_off'], df['diff_rush_off'], df['diff_rz'], df['diff_sack'], df['diff_def_pass'],
        df['diff_turnover'], df['diff_third_down'], df['diff_fg'],
        df['diff_pen']
    ]
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'brown']
    axs_list = axs.flatten()
    for ax, x, c in zip(axs_list, ivs, colors):
        ax.scatter(x, df['home_win_prob'], alpha=0.5, color=c)
        x_clean = x.replace([np.inf, -np.inf], np.nan).dropna()
        y_clean = df['home_win_prob'][x_clean.index]
        if len(x_clean) > 1 and x_clean.std() > 0:
            plot_regression_line(ax, x_clean, y_clean, color='k', ls='-', lw=2)
            r = corrcoeff(x_clean, y_clean)
            ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
    xlabels = (
        'Pass Off EPA', 'Rush Off EPA', 'Red Zone %', 'Sack Rate', 'Def Pass EPA',
        'Turnover Margin', '3rd Down %', 'FG EPA',
        'Penalty Yds'
    )
    [ax.set_xlabel(s) for ax, s in zip(axs_list, xlabels)]
    axs[0, 0].set_ylabel('Home Win Probability')
    axs[1, 0].set_ylabel('Home Win Probability')
    axs[2, 0].set_ylabel('Home Win Probability')
    [ax.set_yticklabels([]) for ax in axs_list if ax not in [axs[0, 0], axs[1, 0], axs[2, 0]]]
    plt.show()