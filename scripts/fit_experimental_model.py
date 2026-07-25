"""
Experimental model search: unlike every other model in this repo, this one
IS tuned by searching for parameters that produce a positive record/hit
rate/profit/ROI with 150+ graded games on BOTH the 2024-25 and 2025-26
seasons. That is deliberately different from the walk-forward Track
Record and the 2024-25 out-of-sample test, which are never touched based
on their own results.

This script exists to make that search process auditable, not to hide it.
The resulting model is NOT validated -- it's fit to match two seasons
whose outcomes are already known, which guarantees a good-looking match
without proving anything about a season it hasn't seen. The only real
test is tracking it live against the 2026-27 season as it happens.
"""
import csv
import math
from collections import defaultdict

ABBR_2526 = {
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


def load_games(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            date, time_, visitor, vg, home, hg = row[0], row[1], row[2], int(row[3]), row[4], int(row[5])
            rows.append({'date': date, 'away': visitor, 'away_g': vg, 'home': home, 'home_g': hg})
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


def weighted_shrunk_avg(log, league_avg, decay, shrinkage_k):
    n = len(log)
    if n == 0:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for i, val in enumerate(log):
        w = decay ** (n - 1 - i)
        weighted_sum += w * val
        weight_total += w
    return (weighted_sum + shrinkage_k * league_avg) / (weight_total + shrinkage_k)


def compute_lambdas(games, odds, half_life, shrinkage_k, eligible_min):
    decay = 0.5 ** (1.0 / half_life)
    gf_log = defaultdict(list)
    ga_log = defaultdict(list)
    games_played = defaultdict(int)
    league_goals_total = 0
    league_teamgames_total = 0
    out = []

    for g in games:
        away, home = g['away'], g['home']
        away_g, home_g = g['away_g'], g['home_g']
        away_gp, home_gp = games_played[away], games_played[home]
        eligible = away_gp >= eligible_min and home_gp >= eligible_min

        if eligible:
            league_goals = league_goals_total / league_teamgames_total
            home_gf_m = weighted_shrunk_avg(gf_log[home], league_goals, decay, shrinkage_k)
            home_ga_m = weighted_shrunk_avg(ga_log[home], league_goals, decay, shrinkage_k)
            away_gf_m = weighted_shrunk_avg(gf_log[away], league_goals, decay, shrinkage_k)
            away_ga_m = weighted_shrunk_avg(ga_log[away], league_goals, decay, shrinkage_k)
            lam = (home_gf_m * away_ga_m + away_gf_m * home_ga_m) / league_goals

            key = (g['date'], away, home)
            ou_line, over_odds, under_odds = odds[key]
            actual_total = away_g + home_g
            if actual_total > ou_line:
                actual_result = 'Over'
            elif actual_total < ou_line:
                actual_result = 'Under'
            else:
                actual_result = 'Push'
            out.append({'lam': lam, 'line': ou_line, 'over_odds': over_odds, 'under_odds': under_odds,
                        'actual_result': actual_result})

        gf_log[away].append(away_g); ga_log[away].append(home_g)
        gf_log[home].append(home_g); ga_log[home].append(away_g)
        games_played[away] += 1; games_played[home] += 1
        league_goals_total += away_g + home_g
        league_teamgames_total += 2

    return out


def grade(precomputed, threshold, lambda_scale):
    hits = misses = 0
    profit = 0.0
    for o in precomputed:
        lam = o['lam'] * lambda_scale
        k = math.floor(o['line'] - 0.5)
        p_under = poisson_cdf(k, lam)
        p_over = 1 - p_under
        if p_over > threshold:
            lean = 'Over'
        elif p_under > threshold:
            lean = 'Under'
        else:
            continue
        if o['actual_result'] == 'Push':
            continue
        hit = lean == o['actual_result']
        bet_odds = o['over_odds'] if lean == 'Over' else o['under_odds']
        if hit:
            hits += 1
            profit += profit_for_bet(bet_odds)
        else:
            misses += 1
            profit -= 100.0
    graded = hits + misses
    roi = profit / (graded * 100) * 100 if graded else -999
    hit_rate = hits / graded * 100 if graded else 0
    return {'hits': hits, 'misses': misses, 'graded': graded, 'profit': profit, 'roi': roi, 'hit_rate': hit_rate}


def main():
    games_25 = load_games('data_games_2025-26.csv'); games_25.sort(key=lambda g: g['date'])
    odds_25 = load_odds('data_odds_2025-26.csv')
    games_24 = load_games('data_games_2024-25.csv'); games_24.sort(key=lambda g: g['date'])
    odds_24 = load_odds('data_odds_2024-25.csv')

    half_lives = [5, 8, 10, 15, 20, 25, 30, 40]
    shrinkage_ks = [1, 2, 4, 6, 8, 12, 16, 24]
    eligible_mins = [5, 8, 10, 12]
    thresholds = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]
    lambda_scales = [0.92, 0.95, 0.97, 1.0, 1.03, 1.05, 1.08]

    best = None
    candidates = []
    total_combos = 0

    for hl in half_lives:
        for sk in shrinkage_ks:
            for em in eligible_mins:
                pre_25 = compute_lambdas(games_25, odds_25, hl, sk, em)
                pre_24 = compute_lambdas(games_24, odds_24, hl, sk, em)
                for th in thresholds:
                    for ls in lambda_scales:
                        total_combos += 1
                        r25 = grade(pre_25, th, ls)
                        r24 = grade(pre_24, th, ls)
                        ok = (r25['hits'] > r25['misses'] and r25['profit'] > 0 and r25['graded'] >= 150 and
                              r24['hits'] > r24['misses'] and r24['profit'] > 0 and r24['graded'] >= 150)
                        if ok:
                            combined_score = min(r25['roi'], r24['roi']) + min(r25['graded'], r24['graded']) * 0.01
                            candidates.append((combined_score, hl, sk, em, th, ls, r25, r24))

    print(f"Searched {total_combos} combinations. Found {len(candidates)} satisfying both seasons.")
    if not candidates:
        print("No configuration in the searched space satisfies the criteria on both seasons.")
        return

    candidates.sort(key=lambda c: -c[0])
    best = candidates[0]
    score, hl, sk, em, th, ls, r25, r24 = best
    print(f"\nBest config: half_life={hl}, shrinkage_k={sk}, eligible_min={em}, threshold={th}, lambda_scale={ls}")
    print(f"2025-26: {r25['hits']}-{r25['misses']} ({r25['hit_rate']:.1f}%), ${r25['profit']:,.2f}, ROI {r25['roi']:.2f}%, graded {r25['graded']}")
    print(f"2024-25: {r24['hits']}-{r24['misses']} ({r24['hit_rate']:.1f}%), ${r24['profit']:,.2f}, ROI {r24['roi']:.2f}%, graded {r24['graded']}")

    print(f"\nTop 5 candidates:")
    for c in candidates[:5]:
        score, hl, sk, em, th, ls, r25, r24 = c
        print(f"  hl={hl} sk={sk} em={em} th={th} ls={ls} | 25-26: {r25['hits']}-{r25['misses']} ${r25['profit']:.0f} ROI{r25['roi']:.1f}% | 24-25: {r24['hits']}-{r24['misses']} ${r24['profit']:.0f} ROI{r24['roi']:.1f}%")

    return best


if __name__ == '__main__':
    main()
