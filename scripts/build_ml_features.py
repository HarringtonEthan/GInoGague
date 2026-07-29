"""
Build a walk-forward-safe feature table for both seasons, combining:
  - team-level recency-weighted GF/GA (existing formula's inputs)
  - home/away split recency-weighted GF/GA
  - rest days for both teams, back-to-back flags
  - starting goalie quality (season SV%, shrunk toward prior season / league avg)
  - injury severity score (from real injury-status data, roster players only)
  - the market's own total line, and its vig-implied over/under probabilities
  - games-into-season (early vs late season)

Every feature for a given game uses ONLY information available before that
game (goalie/injury data is already "pre-game" in the source; GF/GA/rest
are computed the same walk-forward way as every other model in this repo).

Output: one row per game per season, written to CSV for the modeling script.
"""
import csv
import math
import datetime
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
    'Tampa Bay Lightning':'TBL','Toronto Maple Leafs':'TOR','Utah Mammoth':'UTA','Utah Hockey Club':'UTA',
    'Vancouver Canucks':'VAN','Vegas Golden Knights':'VGK','Winnipeg Jets':'WPG',
    'Washington Capitals':'WSH'
}

HALF_LIFE = 15
DECAY = 0.5 ** (1.0 / HALF_LIFE)
SHRINKAGE_K = 12
ELIGIBLE_MIN = 8
LEAGUE_SV_DEFAULT = 0.905
SEVERITY = {'out for season': 3, 'on IR': 2, 'out indefinitely': 2, 'day-to-day': 1,
            'non-roster (injured)': 1, 'unspecified': 1}


def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100) * 100
    return abs(odds) / (abs(odds) + 100) * 100


def wsa(log, avg):
    n = len(log)
    if n == 0:
        return None
    ws, wt = 0.0, 0.0
    for i, val in enumerate(log):
        w = DECAY ** (n - 1 - i)
        ws += w * val
        wt += w
    return (ws + SHRINKAGE_K * avg) / (wt + SHRINKAGE_K)


def _ordinal(d):
    y, m, dd = (int(x) for x in d.split('-'))
    return datetime.date(y, m, dd).toordinal()


def load_games(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            date, time_, away, ag, home, hg = row[0], row[1], row[2], int(row[3]), row[4], int(row[5])
            rows.append({'date': date, 'away': away, 'away_g': ag, 'home': home, 'home_g': hg})
    return rows


def load_odds(path):
    odds = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            date, away, home = row[0], row[3], row[5]
            ou_line, over_odds, under_odds = float(row[10]), int(row[11]), int(row[12])
            odds[(date, away, home)] = (ou_line, over_odds, under_odds)
    return odds


def load_goalie_data(path):
    by_key = {}
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            if row['game_type'] != 'REG':
                continue
            by_key[(row['season'], row['date'], row['team'])] = row
    return by_key


def load_injury_severity(path):
    sev = defaultdict(float)
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            if row['game_type'] != 'REG' or row['nhl_roster_player'] != 'Y':
                continue
            key = (row['season'], row['date'], row['team'])
            sev[key] += SEVERITY.get(row['status'], 1)
    return sev


def goalie_quality(row):
    if row is None:
        return LEAGUE_SV_DEFAULT
    pre_gp = int(row['pre_gp']) if row['pre_gp'] else 0
    pre_sv = float(row['pre_sv_pct']) if row['pre_sv_pct'] else None
    prior_sv = float(row['prior_season_sv_pct']) if row['prior_season_sv_pct'] else None
    prior_gp = int(row['prior_season_gp']) if row['prior_season_gp'] else 0
    if pre_sv is not None and pre_gp > 0:
        prior_component = prior_sv if (prior_sv is not None and prior_gp > 0) else LEAGUE_SV_DEFAULT
        w_current = min(pre_gp, 20)
        w_prior = 10
        return (pre_sv * w_current + prior_component * w_prior) / (w_current + w_prior)
    if prior_sv is not None and prior_gp > 0:
        return (prior_sv * prior_gp + LEAGUE_SV_DEFAULT * 10) / (prior_gp + 10)
    return LEAGUE_SV_DEFAULT


def build(season_label, games_path, odds_path, goalie_by_key, sev_by_key):
    games = load_games(games_path)
    games.sort(key=lambda g: g['date'])
    odds = load_odds(odds_path)

    gf_log = defaultdict(list); ga_log = defaultdict(list)
    gf_home_log = defaultdict(list); ga_home_log = defaultdict(list)
    gf_away_log = defaultdict(list); ga_away_log = defaultdict(list)
    games_played = defaultdict(int)
    last_date = {}
    league_goals_total = 0; league_teamgames_total = 0
    league_home_goals = 0; league_home_n = 0
    league_away_goals = 0; league_away_n = 0

    rows_out = []
    game_num = 0

    for g in games:
        game_num += 1
        away, home = g['away'], g['home']
        away_g, home_g = g['away_g'], g['home_g']
        away_gp, home_gp = games_played[away], games_played[home]
        eligible = away_gp >= ELIGIBLE_MIN and home_gp >= ELIGIBLE_MIN
        d_ord = _ordinal(g['date'])

        if eligible:
            league_avg = league_goals_total / league_teamgames_total
            league_home_avg = league_home_goals / league_home_n
            league_away_avg = league_away_goals / league_away_n

            home_gf = wsa(gf_log[home], league_avg); home_ga = wsa(ga_log[home], league_avg)
            away_gf = wsa(gf_log[away], league_avg); away_ga = wsa(ga_log[away], league_avg)
            home_gf_h = wsa(gf_home_log[home], league_home_avg); home_ga_h = wsa(ga_home_log[home], league_away_avg)
            away_gf_a = wsa(gf_away_log[away], league_away_avg); away_ga_a = wsa(ga_away_log[away], league_home_avg)

            home_rest = min(d_ord - last_date[home] - 1, 5) if home in last_date else 5
            away_rest = min(d_ord - last_date[away] - 1, 5) if away in last_date else 5

            home_goalie_row = goalie_by_key.get((season_label, g['date'], ABBR[home]))
            away_goalie_row = goalie_by_key.get((season_label, g['date'], ABBR[away]))
            home_goalie_sv = goalie_quality(home_goalie_row)
            away_goalie_sv = goalie_quality(away_goalie_row)

            home_sev = sev_by_key.get((season_label, g['date'], ABBR[home]), 0.0)
            away_sev = sev_by_key.get((season_label, g['date'], ABBR[away]), 0.0)

            key = (g['date'], away, home)
            if key not in odds:
                games_played[away] += 1; games_played[home] += 1
                gf_log[away].append(away_g); ga_log[away].append(home_g)
                gf_log[home].append(home_g); ga_log[home].append(away_g)
                gf_away_log[away].append(away_g); ga_away_log[away].append(home_g)
                gf_home_log[home].append(home_g); ga_home_log[home].append(away_g)
                league_goals_total += away_g + home_g; league_teamgames_total += 2
                league_home_goals += home_g; league_home_n += 1
                league_away_goals += away_g; league_away_n += 1
                last_date[away] = d_ord; last_date[home] = d_ord
                continue
            ou_line, over_odds, under_odds = odds[key]
            implied_over = american_to_implied(over_odds)
            implied_under = american_to_implied(under_odds)

            actual_total = home_g + away_g
            if actual_total > ou_line: label = 1
            elif actual_total < ou_line: label = 0
            else: label = None  # push, excluded from training/eval

            rows_out.append({
                'season': season_label, 'date': g['date'], 'away': ABBR[away], 'home': ABBR[home],
                'game_num': game_num,
                'home_gf': home_gf, 'home_ga': home_ga, 'away_gf': away_gf, 'away_ga': away_ga,
                'home_gf_home': home_gf_h, 'home_ga_home': home_ga_h,
                'away_gf_away': away_gf_a, 'away_ga_away': away_ga_a,
                'home_rest': home_rest, 'away_rest': away_rest,
                'home_b2b': 1 if home_rest <= 0 else 0, 'away_b2b': 1 if away_rest <= 0 else 0,
                'home_goalie_sv': home_goalie_sv, 'away_goalie_sv': away_goalie_sv,
                'home_severity': home_sev, 'away_severity': away_sev,
                'line': ou_line, 'implied_over': implied_over, 'implied_under': implied_under,
                'over_odds': over_odds, 'under_odds': under_odds,
                'actual_total': actual_total, 'label': label,
            })

        gf_log[away].append(away_g); ga_log[away].append(home_g)
        gf_log[home].append(home_g); ga_log[home].append(away_g)
        gf_away_log[away].append(away_g); ga_away_log[away].append(home_g)
        gf_home_log[home].append(home_g); ga_home_log[home].append(away_g)
        games_played[away] += 1; games_played[home] += 1
        league_goals_total += away_g + home_g; league_teamgames_total += 2
        league_home_goals += home_g; league_home_n += 1
        league_away_goals += away_g; league_away_n += 1
        last_date[away] = d_ord; last_date[home] = d_ord

    return rows_out


def main():
    GOA = "/root/.claude/uploads/05af2cab-2153-586e-b0a3-45ddc0731d30/e154b18c-nhl_games_goalies_202425_202526.csv"
    INJ = "/root/.claude/uploads/05af2cab-2153-586e-b0a3-45ddc0731d30/2bf360bc-nhl_pregame_injury_reports_202425_202526.csv"
    goalie_by_key = load_goalie_data(GOA)
    sev_by_key = load_injury_severity(INJ)

    all_rows = []
    all_rows += build('2024-25', 'data_games_2024-25.csv', 'data_odds_2024-25.csv', goalie_by_key, sev_by_key)
    all_rows += build('2025-26', 'data_games_2025-26.csv', 'data_odds_2025-26.csv', goalie_by_key, sev_by_key)

    fields = list(all_rows[0].keys())
    with open('ml_features.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"wrote {len(all_rows)} rows to ml_features.csv")
    print(f"  2024-25: {sum(1 for r in all_rows if r['season']=='2024-25')}")
    print(f"  2025-26: {sum(1 for r in all_rows if r['season']=='2025-26')}")
    print(f"  with labels (non-push): {sum(1 for r in all_rows if r['label'] is not None)}")


if __name__ == '__main__':
    main()
