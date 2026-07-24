"""
Step 1 of the rest-day enhancement: measure whether rest actually predicts
goals in the data we have, before deciding to use it.

For every eligible team-game in both seasons, compute:
  - days of rest since that team's previous game (capped, back-to-back = 0)
  - the team's expected goals from the existing (defense-aware + recency +
    shrinkage) walk-forward model
  - the ratio of actual goals scored to expected goals

Then group by rest bucket and look at the average ratio. This is a
calibration check, not a target search -- the question is "does this
signal exist and how big is it", not "what value makes the record look
good".
"""
import csv
import math
from collections import defaultdict

ELIGIBLE_MIN_GAMES = 10
DECAY_HALF_LIFE = 15
DECAY = 0.5 ** (1.0 / DECAY_HALF_LIFE)
SHRINKAGE_K = 8


def weighted_shrunk_avg(log, league_avg):
    n = len(log)
    if n == 0:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for i, val in enumerate(log):
        w = DECAY ** (n - 1 - i)
        weighted_sum += w * val
        weight_total += w
    return (weighted_sum + SHRINKAGE_K * league_avg) / (weight_total + SHRINKAGE_K)


def load_games(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            date, time_, visitor, vg, home, hg = row[0], row[1], row[2], int(row[3]), row[4], int(row[5])
            rows.append({'date': date, 'time': time_, 'away': visitor, 'away_g': vg, 'home': home, 'home_g': hg})
    return rows


def date_to_ordinal(d):
    y, m, dd = (int(x) for x in d.split('-'))
    return (y, m, dd).__hash__() if False else _ordinal(y, m, dd)


def _ordinal(y, m, d):
    import datetime
    return datetime.date(y, m, d).toordinal()


def analyze(games_path, season_label):
    games = load_games(games_path)
    games.sort(key=lambda g: g['date'])

    gf_log = defaultdict(list)
    ga_log = defaultdict(list)
    games_played = defaultdict(int)
    last_game_date = {}
    league_goals_total = 0
    league_teamgames_total = 0

    bucket_data = defaultdict(list)  # rest_bucket -> list of (actual, expected)

    for g in games:
        away, home = g['away'], g['home']
        away_g, home_g = g['away_g'], g['home_g']
        away_gp, home_gp = games_played[away], games_played[home]
        eligible = away_gp >= ELIGIBLE_MIN_GAMES and home_gp >= ELIGIBLE_MIN_GAMES
        d_ord = _ordinal(*[int(x) for x in g['date'].split('-')])

        if eligible:
            league_goals = league_goals_total / league_teamgames_total
            home_gf_m = weighted_shrunk_avg(gf_log[home], league_goals)
            home_ga_m = weighted_shrunk_avg(ga_log[home], league_goals)
            away_gf_m = weighted_shrunk_avg(gf_log[away], league_goals)
            away_ga_m = weighted_shrunk_avg(ga_log[away], league_goals)

            exp_home = home_gf_m * away_ga_m / league_goals
            exp_away = away_gf_m * home_ga_m / league_goals

            for team, exp_g, actual_g in [(home, exp_home, home_g), (away, exp_away, away_g)]:
                if team in last_game_date:
                    rest = d_ord - last_game_date[team] - 1  # days off between games
                    rest = max(0, min(rest, 4))
                    if exp_g > 0:
                        bucket_data[rest].append((actual_g, exp_g))

        gf_log[away].append(away_g); ga_log[away].append(home_g)
        gf_log[home].append(home_g); ga_log[home].append(away_g)
        games_played[away] += 1; games_played[home] += 1
        league_goals_total += away_g + home_g
        league_teamgames_total += 2
        last_game_date[away] = d_ord
        last_game_date[home] = d_ord

    print(f"\n=== {season_label} ===")
    print(f"{'Rest days':<12}{'N':>6}{'Avg actual':>12}{'Avg expected':>14}{'Ratio':>10}")
    overall_n = 0
    for rest in sorted(bucket_data):
        vals = bucket_data[rest]
        n = len(vals)
        overall_n += n
        avg_act = sum(a for a, e in vals) / n
        avg_exp = sum(e for a, e in vals) / n
        ratio = avg_act / avg_exp
        label = f"{rest}d (B2B)" if rest == 0 else f"{rest}d"
        print(f"{label:<12}{n:>6}{avg_act:>12.3f}{avg_exp:>14.3f}{ratio:>10.4f}")
    print(f"Total team-games: {overall_n}")
    return bucket_data


if __name__ == '__main__':
    b1 = analyze('data_games_2024-25.csv', '2024-25')
    b2 = analyze('data_games_2025-26.csv', '2025-26')
