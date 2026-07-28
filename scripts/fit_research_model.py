"""
Research model search: same "search until it fits both known seasons"
approach as fit_experimental_model.py, but with a richer formula
grounded in things actually verified against real data first:

  - Home-ice advantage: well-documented in outside research (~0.25-0.28
    extra goals/game for home teams, statistically significant) AND
    confirmed directly in both of our own seasons (+0.319 goals in
    2024-25, +0.130 in 2025-26). Modeled here as team-specific,
    walk-forward, recency-weighted home/away scoring splits -- not a
    single flat league constant, so a team's own home/away tendency
    matters, not just the league's.
  - Divisional-rivalry scoring effect: searched for and found only weak,
    non-NHL-specific blog claims. Tested directly against our own data:
    -0.095 goals in 2024-25, -0.006 in 2025-26 (416 games each) -- not a
    stable, real effect. NOT included.
  - Back-to-back fatigue: real, but confirmed to be essentially the same
    relative penalty for home and road teams once the home/road baseline
    is controlled for (~93-94% of expected in both seasons, both sides).
    Kept as an optional, searchable multiplier since it's real signal,
    even though the standalone experimental-model test found it doesn't
    reliably help betting performance once the market has priced it in.

Like fit_experimental_model.py, this DOES search for parameters that
produce a winning record on both known seasons -- that's disclosed
plainly on the site, and the only real test is live 2026-27 tracking.
"""
import csv
import math
import datetime
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
    'Tampa Bay Lightning':'TBL','Toronto Maple Leafs':'TOR','Utah Mammoth':'UTA','Utah Hockey Club':'UTA',
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


def _ordinal(d):
    y, m, dd = (int(x) for x in d.split('-'))
    return datetime.date(y, m, dd).toordinal()


def weighted_shrunk_avg(log, prior, decay, shrinkage_k):
    n = len(log)
    if n == 0:
        return None
    ws, wt = 0.0, 0.0
    for i, val in enumerate(log):
        w = decay ** (n - 1 - i)
        ws += w * val
        wt += w
    return (ws + shrinkage_k * prior) / (wt + shrinkage_k)


def compute_games(games, odds, half_life, shrinkage_k, eligible_min, b2b_mult):
    decay = 0.5 ** (1.0 / half_life)
    gf_home_log = defaultdict(list); ga_home_log = defaultdict(list)
    gf_away_log = defaultdict(list); ga_away_log = defaultdict(list)
    home_games_played = defaultdict(int); away_games_played = defaultdict(int)
    last_date = {}
    league_home_goals_total = 0; league_home_games = 0
    league_away_goals_total = 0; league_away_games = 0
    out = []

    for g in games:
        away, home = g['away'], g['home']
        away_g, home_g = g['away_g'], g['home_g']
        home_gp = home_games_played[home]
        away_gp = away_games_played[away]
        eligible = home_gp >= eligible_min and away_gp >= eligible_min
        d_ord = _ordinal(g['date'])

        if eligible:
            league_home_avg = league_home_goals_total / league_home_games
            league_away_avg = league_away_goals_total / league_away_games
            league_avg = (league_home_avg + league_away_avg) / 2

            home_gf = weighted_shrunk_avg(gf_home_log[home], league_home_avg, decay, shrinkage_k)
            home_ga = weighted_shrunk_avg(ga_home_log[home], league_away_avg, decay, shrinkage_k)
            away_gf = weighted_shrunk_avg(gf_away_log[away], league_away_avg, decay, shrinkage_k)
            away_ga = weighted_shrunk_avg(ga_away_log[away], league_home_avg, decay, shrinkage_k)

            exp_home = home_gf * away_ga / league_avg
            exp_away = away_gf * home_ga / league_avg

            if b2b_mult != 1.0:
                if home in last_date and (d_ord - last_date[home] - 1) <= 0:
                    exp_home *= b2b_mult
                if away in last_date and (d_ord - last_date[away] - 1) <= 0:
                    exp_away *= b2b_mult

            lam = exp_home + exp_away

            key = (g['date'], away, home)
            ou_line, over_odds, under_odds = odds[key]
            actual_total = away_g + home_g
            if actual_total > ou_line: actual_result = 'Over'
            elif actual_total < ou_line: actual_result = 'Under'
            else: actual_result = 'Push'

            out.append({'date': g['date'], 'lam': lam, 'line': ou_line, 'over_odds': over_odds,
                        'under_odds': under_odds, 'actual_result': actual_result})

        gf_home_log[home].append(home_g); ga_home_log[home].append(away_g)
        gf_away_log[away].append(away_g); ga_away_log[away].append(home_g)
        home_games_played[home] += 1; away_games_played[away] += 1
        league_home_goals_total += home_g; league_home_games += 1
        league_away_goals_total += away_g; league_away_games += 1
        last_date[home] = d_ord; last_date[away] = d_ord

    return out


def grade(precomputed, threshold):
    hits = misses = 0
    profit = 0.0
    for o in precomputed:
        k = math.floor(o['line'] - 0.5)
        p_under = poisson_cdf(k, o['lam'])
        p_over = 1 - p_under
        if p_over > threshold: lean = 'Over'
        elif p_under > threshold: lean = 'Under'
        else: continue
        if o['actual_result'] == 'Push': continue
        hit = lean == o['actual_result']
        bet_odds = o['over_odds'] if lean == 'Over' else o['under_odds']
        if hit:
            hits += 1; profit += profit_for_bet(bet_odds)
        else:
            misses += 1; profit -= 100.0
    graded = hits + misses
    roi = profit / (graded * 100) * 100 if graded else -999
    hit_rate = hits / graded * 100 if graded else 0
    return {'hits': hits, 'misses': misses, 'graded': graded, 'profit': profit, 'roi': roi, 'hit_rate': hit_rate}


def main():
    games_25 = load_games('data_games_2025-26.csv'); games_25.sort(key=lambda g: g['date'])
    odds_25 = load_odds('data_odds_2025-26.csv')
    games_24 = load_games('data_games_2024-25.csv'); games_24.sort(key=lambda g: g['date'])
    odds_24 = load_odds('data_odds_2024-25.csv')

    half_lives = [8, 10, 12, 15, 20, 25, 30]
    shrinkage_ks = [2, 4, 6, 8, 10, 12, 16]
    eligible_mins = [5, 8, 10]
    thresholds = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56]
    b2b_mults = [1.0, 0.97, 0.95, 0.93]

    candidates = []
    total = 0
    for hl in half_lives:
        for sk in shrinkage_ks:
            for em in eligible_mins:
                for bm in b2b_mults:
                    pre_25 = compute_games(games_25, odds_25, hl, sk, em, bm)
                    pre_24 = compute_games(games_24, odds_24, hl, sk, em, bm)
                    for th in thresholds:
                        total += 1
                        r25 = grade(pre_25, th)
                        r24 = grade(pre_24, th)
                        ok = (r25['hit_rate'] >= 60.0 and r25['profit'] > 0 and r25['graded'] >= 280 and
                              r24['hit_rate'] >= 60.0 and r24['profit'] > 0 and r24['graded'] >= 280)
                        if ok:
                            score = min(r25['graded'], r24['graded']) + min(r25['roi'], r24['roi'])
                            candidates.append((score, hl, sk, em, bm, th, r25, r24))

    print(f"Searched {total} combinations. {len(candidates)} satisfy 60%+ hit rate, positive profit, 280+ graded on BOTH seasons.")
    if not candidates:
        print("None found in this space.")
        return
    candidates.sort(key=lambda c: -c[0])
    for c in candidates[:8]:
        score, hl, sk, em, bm, th, r25, r24 = c
        print(f"hl={hl} sk={sk} em={em} b2b={bm} th={th} | "
              f"25-26: {r25['hits']}-{r25['misses']} ({r25['hit_rate']:.1f}%) ${r25['profit']:.0f} ROI{r25['roi']:.1f}% n={r25['graded']} | "
              f"24-25: {r24['hits']}-{r24['misses']} ({r24['hit_rate']:.1f}%) ${r24['profit']:.0f} ROI{r24['roi']:.1f}% n={r24['graded']}")


if __name__ == '__main__':
    main()
