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
    'Tampa Bay Lightning':'TBL','Toronto Maple Leafs':'TOR','Utah Mammoth':'UTA',
    'Vancouver Canucks':'VAN','Vegas Golden Knights':'VGK','Winnipeg Jets':'WPG',
    'Washington Capitals':'WSH'
}

ELIGIBLE_MIN_GAMES = 10
LEAN_THRESHOLD = 0.60

# Decided up front, from sports-analytics convention -- NOT tuned against
# this dataset's results. Changing either of these after seeing the score
# would be exactly the p-hacking this project is trying to avoid.
DECAY_HALF_LIFE = 15   # games; a team's scoring rate from ~3 weeks ago counts half as much as tonight
DECAY = 0.5 ** (1.0 / DECAY_HALF_LIFE)
SHRINKAGE_K = 8         # pseudo-games of league-average prior blended into small samples


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
    """log: list of stat values, oldest first. Exponential recency decay,
    then Bayesian shrinkage toward league_avg with SHRINKAGE_K pseudo-games."""
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
    games = load_games_raw('games_2025-26.csv')
    games.sort(key=lambda g: g['date'])
    odds = load_odds('odds_2025-26.csv')

    gf_log = defaultdict(list)
    ga_log = defaultdict(list)
    cum_gf = defaultdict(int)
    cum_ga = defaultdict(int)
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
            'away_gp_before': away_gp, 'home_gp_before': home_gp,
            # raw, unweighted cumulative averages -- kept for transparency/audit
            'away_gf_avg': (cum_gf[away] / away_gp) if away_gp else None,
            'away_ga_avg': (cum_ga[away] / away_gp) if away_gp else None,
            'home_gf_avg': (cum_gf[home] / home_gp) if home_gp else None,
            'home_ga_avg': (cum_ga[home] / home_gp) if home_gp else None,
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

            edge_over = p_over * 100 - american_to_implied(over_odds)
            edge_under = p_under * 100 - american_to_implied(under_odds)

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
                'home_gf_model': home_gf_m, 'home_ga_model': home_ga_m,
                'away_gf_model': away_gf_m, 'away_ga_model': away_ga_m,
                'p_over': p_over, 'p_under': p_under, 'lean': lean,
                'edge_over': edge_over, 'edge_under': edge_under,
                'actual_result': actual_result, 'graded': graded, 'hit': hit, 'profit': profit,
            })
        else:
            row.update({'lambda': None, 'line': None, 'over_odds': None, 'under_odds': None,
                        'home_gf_model': None, 'home_ga_model': None, 'away_gf_model': None, 'away_ga_model': None,
                        'p_over': None, 'p_under': None, 'lean': None, 'edge_over': None,
                        'edge_under': None, 'actual_result': None, 'graded': False, 'hit': None,
                        'profit': None})

        output.append(row)

        gf_log[away].append(away_g); ga_log[away].append(home_g)
        gf_log[home].append(home_g); ga_log[home].append(away_g)
        cum_gf[away] += away_g; cum_ga[away] += home_g
        cum_gf[home] += home_g; cum_ga[home] += away_g
        games_played[away] += 1; games_played[home] += 1
        league_goals_total += away_g + home_g
        league_teamgames_total += 2

    with open('full_backtest_output.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Date','Away','Away_Abbr','Home','Home_Abbr','Away_G','Home_G','Total',
                    'Away_GP_Before','Home_GP_Before','Away_GF_Avg','Away_GA_Avg','Home_GF_Avg','Home_GA_Avg',
                    'Eligible','Lambda','Line','Over_Odds','Under_Odds',
                    'Home_GF_Model','Home_GA_Model','Away_GF_Model','Away_GA_Model',
                    'P_Over','P_Under','Lean','Edge_Over','Edge_Under','Actual_Result','Graded','Hit','Profit'])
        for o in output:
            def fmt(v, nd=4):
                return f"{v:.{nd}f}" if isinstance(v, float) else ('' if v is None else v)
            w.writerow([o['date'], o['away'], o['away_abbr'], o['home'], o['home_abbr'],
                        o['away_g'], o['home_g'], o['total'], o['away_gp_before'], o['home_gp_before'],
                        fmt(o['away_gf_avg']), fmt(o['away_ga_avg']), fmt(o['home_gf_avg']), fmt(o['home_ga_avg']),
                        o['eligible'], fmt(o['lambda']), fmt(o['line'], 1), o['over_odds'], o['under_odds'],
                        fmt(o['home_gf_model']), fmt(o['home_ga_model']), fmt(o['away_gf_model']), fmt(o['away_ga_model']),
                        fmt(o['p_over']), fmt(o['p_under']), o['lean'], fmt(o['edge_over'], 2), fmt(o['edge_under'], 2),
                        o['actual_result'], o['graded'], o['hit'], fmt(o['profit'], 2)])

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
            graded_rows.append((hit, profit))

        hits = sum(1 for h, p in graded_rows if h)
        misses = len(graded_rows) - hits
        total_profit = sum(p for h, p in graded_rows)
        total_staked = len(graded_rows) * 100.0
        print(f"\n=== Threshold {threshold_label} ===")
        print(f"Graded: {len(graded_rows)} ({hits}-{misses})  Toss-ups: {tossups}")
        if graded_rows:
            print(f"Hit rate: {hits/len(graded_rows)*100:.2f}%")
            print(f"Profit: ${total_profit:,.2f}  ROI: {total_profit/total_staked*100:.2f}%")


if __name__ == '__main__':
    main()
