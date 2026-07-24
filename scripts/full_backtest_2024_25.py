"""
Out-of-sample validation: the SAME model, with the SAME parameters, that
was fixed before ever seeing the 2025-26 season, run against the 2024-25
season instead. Nothing here is tuned to 2024-25 results -- every constant
below is copy-identical to full_backtest.py.

This is a genuine out-of-sample test: the model's design (defense-aware
formula, recency half-life, shrinkage strength, 60% lean threshold) was
all decided using 2025-26 data (and sports-analytics convention) months
before this 2024-25 dataset was ever touched.
"""
import csv
import math
from collections import defaultdict

ABBR = {
    'Anaheim Ducks':'ANA','Boston Bruins':'BOS','Buffalo Sabres':'BUF','Calgary Flames':'CGY',
    'Carolina Hurricanes':'CAR','Chicago Blackhawks':'CHI','Colorado Avalanche':'COL',
    'Columbus Blue Jackets':'CBJ','Dallas Stars':'DAL','Detroit Red Wings':'DET',
    'Edmonton Oilers':'EDM','Florida Panthers':'FLA','Los Angeles Kings':'LAK',
    'Minnesota Wild':'MIN','Montreal Canadiens':'MTL','Nashville Predators':'NSH',
    'New Jersey Devils':'NJD','New York Islanders':'NYI','New York Rangers':'NYR',
    'Ottawa Senators':'OTT','Philadelphia Flyers':'PHI','Pittsburgh Penguins':'PIT',
    'San Jose Sharks':'SJS','Seattle Kraken':'SEA','St. Louis Blues':'STL',
    'Tampa Bay Lightning':'TBL','Toronto Maple Leafs':'TOR','Utah Hockey Club':'UTA',
    'Vancouver Canucks':'VAN','Vegas Golden Knights':'VGK','Winnipeg Jets':'WPG',
    'Washington Capitals':'WSH'
}

ELIGIBLE_MIN_GAMES = 10
LEAN_THRESHOLD = 0.60

# Identical to full_backtest.py -- not re-tuned for this season.
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


def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100) * 100
    return abs(odds) / (abs(odds) + 100) * 100


def load_games_raw(path):
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


def main():
    games = load_games_raw('data_games_2024-25.csv')
    games.sort(key=lambda g: g['date'])
    odds = load_odds('data_odds_2024-25.csv')

    gf_log = defaultdict(list)
    ga_log = defaultdict(list)
    games_played = defaultdict(int)
    league_goals_total = 0
    league_teamgames_total = 0

    output = []

    for g in games:
        away, home = g['away'], g['home']
        away_g, home_g = g['away_g'], g['home_g']
        away_gp, home_gp = games_played[away], games_played[home]
        eligible = away_gp >= ELIGIBLE_MIN_GAMES and home_gp >= ELIGIBLE_MIN_GAMES

        row = {
            'date': g['date'], 'away': away, 'away_abbr': ABBR[away],
            'home': home, 'home_abbr': ABBR[home],
            'away_g': away_g, 'home_g': home_g, 'total': away_g + home_g,
            'eligible': eligible,
        }

        if eligible:
            league_goals = league_goals_total / league_teamgames_total

            home_gf_m = weighted_shrunk_avg(gf_log[home], league_goals)
            home_ga_m = weighted_shrunk_avg(ga_log[home], league_goals)
            away_gf_m = weighted_shrunk_avg(gf_log[away], league_goals)
            away_ga_m = weighted_shrunk_avg(ga_log[away], league_goals)

            lam = (home_gf_m * away_ga_m + away_gf_m * home_ga_m) / league_goals

            key = (g['date'], away, home)
            ou_line, over_odds, under_odds = odds[key]
            k = math.floor(ou_line - 0.5)
            p_under = poisson_cdf(k, lam)
            p_over = 1 - p_under

            if p_over > LEAN_THRESHOLD:
                lean = 'Over'
            elif p_under > LEAN_THRESHOLD:
                lean = 'Under'
            else:
                lean = 'Toss-up'

            actual_total = row['total']
            if actual_total > ou_line:
                actual_result = 'Over'
            elif actual_total < ou_line:
                actual_result = 'Under'
            else:
                actual_result = 'Push'

            graded = lean in ('Over', 'Under') and actual_result != 'Push'
            hit = graded and (lean == actual_result)

            profit = None
            if graded:
                bet_odds = over_odds if lean == 'Over' else under_odds
                profit = profit_for_bet(bet_odds) if hit else -100.0

            row.update({
                'lambda': lam, 'line': ou_line, 'over_odds': over_odds, 'under_odds': under_odds,
                'p_over': p_over, 'p_under': p_under, 'lean': lean,
                'actual_result': actual_result, 'graded': graded, 'hit': hit, 'profit': profit,
            })
        else:
            row.update({'lambda': None, 'line': None, 'over_odds': None, 'under_odds': None,
                        'p_over': None, 'p_under': None, 'lean': None,
                        'actual_result': None, 'graded': False, 'hit': None, 'profit': None})

        output.append(row)

        gf_log[away].append(away_g); ga_log[away].append(home_g)
        gf_log[home].append(home_g); ga_log[home].append(away_g)
        games_played[away] += 1; games_played[home] += 1
        league_goals_total += away_g + home_g
        league_teamgames_total += 2

    with open('full_backtest_2024-25_output.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Date','Away','Away_Abbr','Home','Home_Abbr','Away_G','Home_G','Total',
                    'Eligible','Lambda','Line','Over_Odds','Under_Odds',
                    'P_Over','P_Under','Lean','Actual_Result','Graded','Hit','Profit'])
        for o in output:
            def fmt(v, nd=4):
                return f"{v:.{nd}f}" if isinstance(v, float) else ('' if v is None else v)
            w.writerow([o['date'], o['away'], o['away_abbr'], o['home'], o['home_abbr'],
                        o['away_g'], o['home_g'], o['total'],
                        o['eligible'], fmt(o['lambda']), fmt(o['line'], 1), o['over_odds'], o['under_odds'],
                        fmt(o['p_over']), fmt(o['p_under']), o['lean'],
                        o['actual_result'], o['graded'], o['hit'], fmt(o['profit'], 2)])

    print(f"Total games: {len(output)}")

    # Split-season robustness check, same convention as 2025-26: chronological halves.
    graded_only = [o for o in output if o['graded']]
    half = len(games) // 2
    half_date = games[half]['date']

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
        misses = len(graded_rows) - hits
        total_profit = sum(p for d, h, p in graded_rows)
        total_staked = len(graded_rows) * 100.0
        print(f"\n=== Threshold {threshold_label} ===")
        print(f"Graded: {len(graded_rows)} ({hits}-{misses})  Toss-ups: {tossups}")
        if graded_rows:
            print(f"Hit rate: {hits/len(graded_rows)*100:.2f}%")
            print(f"Profit: ${total_profit:,.2f}  ROI: {total_profit/total_staked*100:.2f}%")

            first_half = [(h, p) for d, h, p in graded_rows if d < half_date]
            second_half = [(h, p) for d, h, p in graded_rows if d >= half_date]
            for label, half_rows in [('First half', first_half), ('Second half', second_half)]:
                if not half_rows:
                    continue
                hh = sum(1 for h, p in half_rows if h)
                hp = sum(p for h, p in half_rows)
                hs = len(half_rows) * 100.0
                print(f"  {label}: {len(half_rows)} graded, {hh}-{len(half_rows)-hh} ({hh/len(half_rows)*100:.1f}%), "
                      f"${hp:,.2f} ROI {hp/hs*100:.2f}%")


if __name__ == '__main__':
    main()
