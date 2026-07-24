"""
Walk-forward backtest with an added back-to-back rest adjustment on top of
the existing defense-aware + recency + shrinkage model. Same eligibility
rule, same decay/shrinkage constants, same 52%/60% lean thresholds as
full_backtest.py / full_backtest_2024_25.py -- the only change is a
multiplicative discount applied to a team's expected goals when that team
is playing on zero days of rest (confirmed as a real, cross-season-stable
effect via scripts/rest_days_analysis.py, not searched for a target).

Usage: full_backtest_rest.py <games_csv> <odds_csv> <b2b_multiplier> <label>
b2b_multiplier of 1.0 disables the adjustment (baseline model).
"""
import csv
import math
import sys
from collections import defaultdict

ELIGIBLE_MIN_GAMES = 10
LEAN_THRESHOLD = 0.60
DECAY_HALF_LIFE = 15
DECAY = 0.5 ** (1.0 / DECAY_HALF_LIFE)
SHRINKAGE_K = 8


def poisson_cdf(k, lam):
    k = max(0, k)
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term = term * lam / i
        total += term
    return total


def profit_for_bet(odds, staked=100.0):
    if odds > 0:
        return staked * (odds / 100.0)
    return staked * (100.0 / abs(odds))


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


def load_odds(path):
    odds = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            date, visitor, home = row[0], row[3], row[5]
            ou_line, over_odds, under_odds = float(row[10]), int(row[11]), int(row[12])
            odds[(date, visitor, home)] = (ou_line, over_odds, under_odds)
    return odds


def _ordinal(y, m, d):
    import datetime
    return datetime.date(y, m, d).toordinal()


def run(games_path, odds_path, b2b_mult, label):
    games = load_games(games_path)
    games.sort(key=lambda g: g['date'])
    odds = load_odds(odds_path)

    gf_log = defaultdict(list)
    ga_log = defaultdict(list)
    games_played = defaultdict(int)
    last_game_date = {}
    league_goals_total = 0
    league_teamgames_total = 0

    output = []

    for g in games:
        away, home = g['away'], g['home']
        away_g, home_g = g['away_g'], g['home_g']
        away_gp, home_gp = games_played[away], games_played[home]
        eligible = away_gp >= ELIGIBLE_MIN_GAMES and home_gp >= ELIGIBLE_MIN_GAMES
        d_ord = _ordinal(*[int(x) for x in g['date'].split('-')])

        row = {'date': g['date'], 'away': away, 'home': home,
               'away_g': away_g, 'home_g': home_g, 'total': away_g + home_g, 'eligible': eligible}

        if eligible:
            league_goals = league_goals_total / league_teamgames_total
            home_gf_m = weighted_shrunk_avg(gf_log[home], league_goals)
            home_ga_m = weighted_shrunk_avg(ga_log[home], league_goals)
            away_gf_m = weighted_shrunk_avg(gf_log[away], league_goals)
            away_ga_m = weighted_shrunk_avg(ga_log[away], league_goals)

            exp_home = home_gf_m * away_ga_m / league_goals
            exp_away = away_gf_m * home_ga_m / league_goals

            if home in last_game_date and (d_ord - last_game_date[home] - 1) <= 0:
                exp_home *= b2b_mult
            if away in last_game_date and (d_ord - last_game_date[away] - 1) <= 0:
                exp_away *= b2b_mult

            lam = exp_home + exp_away

            key = (g['date'], away, home)
            ou_line, over_odds, under_odds = odds[key]
            k = math.floor(ou_line - 0.5)
            p_under = poisson_cdf(k, lam)
            p_over = 1 - p_under

            actual_total = row['total']
            if actual_total > ou_line:
                actual_result = 'Over'
            elif actual_total < ou_line:
                actual_result = 'Under'
            else:
                actual_result = 'Push'

            row.update({'p_over': p_over, 'p_under': p_under, 'line': ou_line,
                        'over_odds': over_odds, 'under_odds': under_odds, 'actual_result': actual_result})
        else:
            row.update({'p_over': None, 'p_under': None, 'line': None,
                        'over_odds': None, 'under_odds': None, 'actual_result': None})

        output.append(row)

        gf_log[away].append(away_g); ga_log[away].append(home_g)
        gf_log[home].append(home_g); ga_log[home].append(away_g)
        games_played[away] += 1; games_played[home] += 1
        league_goals_total += away_g + home_g
        league_teamgames_total += 2
        last_game_date[away] = d_ord
        last_game_date[home] = d_ord

    print(f"\n########## {label} (b2b_mult={b2b_mult}) ##########")
    half = len(games) // 2
    half_date = games[half]['date']

    results = {}
    for threshold_label, thresh in [('52%', 0.52), ('60%', 0.60)]:
        graded_rows = []
        tossups = 0
        for o in output:
            if not o['eligible']:
                continue
            if o['p_over'] > thresh:
                lean = 'Over'
            elif o['p_under'] > thresh:
                lean = 'Under'
            else:
                lean = 'Toss-up'
            if lean == 'Toss-up':
                tossups += 1
                continue
            if o['actual_result'] == 'Push':
                continue
            hit = lean == o['actual_result']
            bet_odds = o['over_odds'] if lean == 'Over' else o['under_odds']
            profit = profit_for_bet(bet_odds) if hit else -100.0
            graded_rows.append((o['date'], hit, profit))

        hits = sum(1 for d, h, p in graded_rows if h)
        n = len(graded_rows)
        total_profit = sum(p for d, h, p in graded_rows)
        total_staked = n * 100.0
        print(f"\n=== Threshold {threshold_label} ===")
        print(f"Graded: {n} ({hits}-{n-hits})  Toss-ups: {tossups}")
        if n:
            hr = hits / n * 100
            roi = total_profit / total_staked * 100
            print(f"Hit rate: {hr:.2f}%  Profit: ${total_profit:,.2f}  ROI: {roi:.2f}%")
            fh = [(h, p) for d, h, p in graded_rows if d < half_date]
            sh = [(h, p) for d, h, p in graded_rows if d >= half_date]
            for lbl, hr_rows in [('First half', fh), ('Second half', sh)]:
                if not hr_rows:
                    continue
                hh = sum(1 for h, p in hr_rows if h)
                hp = sum(p for h, p in hr_rows)
                print(f"  {lbl}: {len(hr_rows)} graded, {hh}-{len(hr_rows)-hh} ({hh/len(hr_rows)*100:.1f}%), "
                      f"${hp:,.2f} ROI {hp/(len(hr_rows)*100)*100:.2f}%")
            results[threshold_label] = {'n': n, 'hits': hits, 'profit': total_profit, 'roi': roi}
    return results


if __name__ == '__main__':
    games_path, odds_path, mult, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    run(games_path, odds_path, mult, label)
