"""
Refresh the site with every game played so far in the live 2026-27 season.

Drop the season's two data files into the repo root, in the same formats as
the 2024-25 / 2025-26 files already here, then run:

    python3 scripts/update_live_season.py

Inputs
  data_games_2026-27.csv  Hockey-Reference schedule/results export
                          (Date,Time,Visitor,G,Home,G,,Att.,LOG,Notes).
                          Rows with no score yet (games not played) are skipped.
  data_odds_2026-27.csv   Closing totals, same columns as data_odds_2025-26.csv.
                          Optional: without it, only the matchup tool's team
                          form is refreshed and the Experimental Model page is
                          left alone (its picks can't be graded without lines).

Outputs
  data_team_form_2026-27.json        each team's current GF/GA + league average
  experimental_backtest_2026-27.csv  every game, with the model's pick and grade
  index.html                         matchup tool's team GF/GA and league average
  experimental-model.html            the "2026-27 (live)" tab's picks

The model is the Experimental Model exactly as it was locked before the
season (scripts/fit_experimental_model.py's chosen config). Every pick uses
only games played before it, so re-running this as the season goes on only
adds new games -- it never changes a pick that was already made. Do not
change these constants mid-season.
"""
import argparse
import csv
import datetime
import json
import math
import os
import re
import sys
from collections import defaultdict

HALF_LIFE = 15
DECAY = 0.5 ** (1.0 / HALF_LIFE)
SHRINKAGE_K = 12
ELIGIBLE_MIN = 8
LEAN_THRESHOLD = 0.58
LAMBDA_SCALE = 1.0

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

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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


def profit_for_win(odds, staked=100.0):
    if odds > 0:
        return staked * (odds / 100.0)
    return staked * (100.0 / abs(odds))


def weighted_shrunk_avg(log, league_avg):
    n = len(log)
    weighted_sum = 0.0
    weight_total = 0.0
    for i, val in enumerate(log):
        w = DECAY ** (n - 1 - i)
        weighted_sum += w * val
        weight_total += w
    return (weighted_sum + SHRINKAGE_K * league_avg) / (weight_total + SHRINKAGE_K)


def abbr(name, path):
    if name not in ABBR:
        sys.exit(f"{path}: unknown team name {name!r} -- add it to ABBR in {os.path.basename(__file__)}")
    return ABBR[name]


def load_games(path):
    """Completed games only, in date order (CSV order kept within a date)."""
    games = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0] or row[0] == 'Date':
                continue
            if row[3].strip() == '' or row[5].strip() == '':
                continue  # scheduled but not played yet
            games.append({'date': row[0], 'away': row[2], 'away_g': int(row[3]),
                          'home': row[4], 'home_g': int(row[5])})
    for g in games:
        abbr(g['away'], path)
        abbr(g['home'], path)
    games.sort(key=lambda g: g['date'])
    return games


def load_odds(path):
    odds = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0] or row[10].strip() == '':
                continue
            key = (row[0], row[3], row[5])
            scores = None
            if row[4].strip() != '' and row[6].strip() != '':
                scores = (int(row[4]), int(row[6]))
            odds[key] = {'line': float(row[10]), 'over': int(row[11]), 'under': int(row[12]), 'scores': scores}
    return odds


def run_model(games, odds):
    """Walk-forward picks for every game, plus each team's form after the last one."""
    gf_log = defaultdict(list)
    ga_log = defaultdict(list)
    goals_total = 0
    teamgames_total = 0
    rows = []
    missing_odds = []
    score_mismatches = []

    for g in games:
        away, home = g['away'], g['home']
        total = g['away_g'] + g['home_g']
        row = {'date': g['date'], 'away': ABBR[away], 'home': ABBR[home],
               'away_g': g['away_g'], 'home_g': g['home_g'], 'total': total, 'eligible': False}

        o = odds.get((g['date'], away, home)) if odds is not None else None
        if o and o['scores'] and o['scores'] != (g['away_g'], g['home_g']):
            score_mismatches.append(f"{g['date']} {away} @ {home}: games file {g['away_g']}-{g['home_g']}, "
                                    f"odds file {o['scores'][0]}-{o['scores'][1]}")

        if len(gf_log[away]) >= ELIGIBLE_MIN and len(gf_log[home]) >= ELIGIBLE_MIN and odds is not None:
            if o is None:
                missing_odds.append(f"{g['date']} {away} @ {home}")
            else:
                league = goals_total / teamgames_total
                home_gf = weighted_shrunk_avg(gf_log[home], league)
                home_ga = weighted_shrunk_avg(ga_log[home], league)
                away_gf = weighted_shrunk_avg(gf_log[away], league)
                away_ga = weighted_shrunk_avg(ga_log[away], league)
                lam = (home_gf * away_ga + away_gf * home_ga) / league * LAMBDA_SCALE
                p_under = poisson_cdf(math.floor(o['line'] - 0.5), lam)
                p_over = 1 - p_under
                if p_over > LEAN_THRESHOLD:
                    lean = 'Over'
                elif p_under > LEAN_THRESHOLD:
                    lean = 'Under'
                else:
                    lean = 'Toss-up'
                if total > o['line']:
                    actual = 'Over'
                elif total < o['line']:
                    actual = 'Under'
                else:
                    actual = 'Push'
                graded = lean != 'Toss-up' and actual != 'Push'
                hit = graded and lean == actual
                profit = None
                if graded:
                    profit = profit_for_win(o['over'] if lean == 'Over' else o['under']) if hit else -100.0
                row.update({'eligible': True, 'line': o['line'], 'p_over': p_over, 'p_under': p_under,
                            'lean': lean,
                            'edge_over': p_over * 100 - american_to_implied(o['over']),
                            'edge_under': p_under * 100 - american_to_implied(o['under']),
                            'actual': actual, 'graded': graded, 'hit': hit, 'profit': profit})
        rows.append(row)

        gf_log[away].append(g['away_g']); ga_log[away].append(g['home_g'])
        gf_log[home].append(g['home_g']); ga_log[home].append(g['away_g'])
        goals_total += total
        teamgames_total += 2

    league = goals_total / teamgames_total
    form = {}
    for team in dict.fromkeys(ABBR.values()):
        full = [n for n, a in ABBR.items() if a == team and n in gf_log]
        if not full:
            continue
        name = full[0]
        form[team] = {'gf': round(weighted_shrunk_avg(gf_log[name], league), 4),
                      'ga': round(weighted_shrunk_avg(ga_log[name], league), 4),
                      'gp': len(gf_log[name])}
    return rows, form, round(league, 6), missing_odds, score_mismatches


def fmt(x, places):
    return f"{x:.{places}f}"


def write_backtest_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Date', 'Away_Abbr', 'Home_Abbr', 'Away_G', 'Home_G', 'Total', 'Eligible', 'Line',
                    'P_Over', 'P_Under', 'Lean', 'Edge_Over', 'Edge_Under', 'Actual_Result', 'Graded', 'Hit', 'Profit'])
        for r in rows:
            base = [r['date'], r['away'], r['home'], r['away_g'], r['home_g'], r['total'], str(r['eligible'])]
            if not r['eligible']:
                w.writerow(base + ['', '', '', '', '', '', '', 'False', '', ''])
                continue
            w.writerow(base + [fmt(r['line'], 1), fmt(r['p_over'], 4), fmt(r['p_under'], 4), r['lean'],
                               fmt(r['edge_over'], 2), fmt(r['edge_under'], 2), r['actual'],
                               str(r['graded']), str(r['hit']), '' if r['profit'] is None else fmt(r['profit'], 2)])


def js_rows(rows):
    out = []
    for r in rows:
        head = f"    {{date:'{r['date']}', game:'{r['away']}@{r['home']}', total:{r['total']}, "
        if not r['eligible']:
            out.append(head + "line:null,probOver:null,probUnder:null,lean:null,edgeOver:null,edgeUnder:null,"
                              "actual:null,outcome:null,profit:null},")
            continue
        outcome = 'null' if not r['graded'] else ('1' if r['hit'] else '0')
        profit = 'null' if r['profit'] is None else fmt(r['profit'], 2)
        out.append(head + f"line:{fmt(r['line'], 1)},probOver:{fmt(r['p_over'], 4)},probUnder:{fmt(r['p_under'], 4)},"
                          f"lean:'{r['lean']}',edgeOver:{fmt(r['edge_over'], 2)},edgeUnder:{fmt(r['edge_under'], 2)},"
                          f"actual:'{r['actual']}',outcome:{outcome},profit:{profit}}},")
    return out


def pretty_date(iso):
    d = datetime.date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def sub_once(pattern, repl, text, what, flags=0):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        sys.exit(f"Couldn't find {what} to update -- has the page markup changed?")
    return new


def update_index(path, form, league, through):
    with open(path) as f:
        html = f.read()
    for team, v in form.items():
        html = sub_once(r"(\{abbr:'" + team + r"',[^}]*?gf:)[\d.]+(,ga:)[\d.]+",
                        lambda m: f"{m.group(1)}{v['gf']}{m.group(2)}{v['ga']}", html, f"{team} in TEAMS")
    comment = ("  // League-wide average goals per team per game, walk-forward computed\n"
               f"  // across every 2026-27 game played through {through} -- the same\n"
               "  // denominator used by the experimental model. Every team's gf/ga above\n"
               "  // is that same team's recency-weighted (15-game half-life), shrinkage-\n"
               "  // blended (toward this league average, K=12) scoring rate as of its\n"
               "  // latest 2026-27 game. Regenerated by scripts/update_live_season.py.\n"
               f"  var LEAGUE_AVG = {league};")
    html = sub_once(r"  // League-wide average goals per team per game.*?var LEAGUE_AVG = [\d.]+;",
                    lambda m: comment, html, "LEAGUE_AVG", flags=re.S)
    html = sub_once(r"\d{4}–\d{2} team stats[^<]*?via Hockey-Reference\.com\.",
                    f"2026–27 team stats through {pretty_date(through)} via Hockey-Reference.com.",
                    html, "footer stats credit")
    with open(path, 'w') as f:
        f.write(html)


def update_experimental(path, rows, through):
    with open(path) as f:
        html = f.read()
    body = '\n'.join(js_rows(rows))
    html = sub_once(r"  var EXP_2627 = \[.*?\];",
                    lambda m: "  var EXP_2627 = [\n" + body + "\n  ];", html, "EXP_2627", flags=re.S)
    note = ("This is the real test. The model was locked before the 2026-27 season began and is not being "
            "touched now that it's underway — every pick tracked as it happens, using only results known "
            f"before each game was played. Graded through {pretty_date(through)}.")
    html = sub_once(r'(<p class="section-note" id="expNote2627">).*?(</p>)',
                    lambda m: m.group(1) + note + m.group(2), html, "2026-27 tab note", flags=re.S)
    # One-time wording change from "hasn't started" to "underway"; a no-op on later runs.
    html = re.sub(r"The only real test is the 2026-27 season, which hasn't happened yet\..*?(</div>)",
                  lambda m: "The only real test is the 2026-27 season, now underway. Its tab fills in game by "
                            "game, live — untouched, no retuning allowed now that it's begun." + m.group(1),
                  html, count=1, flags=re.S)
    html = re.sub(r"(?<!2026–27, )2025–26 and 2024–25 team stats via Hockey-Reference\.com\.",
                  "2026–27, 2025–26 and 2024–25 team stats via Hockey-Reference.com.", html, count=1)
    with open(path, 'w') as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--season', default='2026-27')
    ap.add_argument('--data-only', action='store_true',
                    help="write the CSV/JSON outputs but don't touch the HTML pages")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    games_path = os.path.join(root, f'data_games_{args.season}.csv')
    odds_path = os.path.join(root, f'data_odds_{args.season}.csv')
    if not os.path.exists(games_path):
        sys.exit(f"Missing {os.path.basename(games_path)} -- export the season's results "
                 "(same format as data_games_2025-26.csv) into the repo root first.")
    games = load_games(games_path)
    if not games:
        sys.exit(f"{os.path.basename(games_path)} has no completed games yet.")
    odds = load_odds(odds_path) if os.path.exists(odds_path) else None

    rows, form, league, missing_odds, mismatches = run_model(games, odds)
    if mismatches:
        print("WARNING: scores disagree between the games and odds files (games file used):")
        for m in mismatches:
            print("  " + m)
    if missing_odds:
        print(f"{len(missing_odds)} eligible game(s) have no line in {os.path.basename(odds_path)}; "
              "add them so every pick can be graded:")
        for m in missing_odds:
            print("  " + m)
        sys.exit(1)

    through = games[-1]['date']
    with open(os.path.join(root, f'data_team_form_{args.season}.json'), 'w') as f:
        json.dump({'league_avg': league, 'through': through,
                   'teams': {t: {'gf': v['gf'], 'ga': v['ga']} for t, v in form.items()}}, f, indent=2)
        f.write('\n')
    if odds is not None:
        write_backtest_csv(rows, os.path.join(root, f'experimental_backtest_{args.season}.csv'))

    graded = [r for r in rows if r['eligible'] and r['graded']]
    hits = sum(r['hit'] for r in graded)
    profit = sum(r['profit'] for r in graded)
    print(f"{args.season}: {len(games)} games through {through}, league avg {league}, {len(form)} teams")
    if odds is None:
        print(f"No {os.path.basename(odds_path)} -- picks not graded, Experimental Model page not updated.")
    elif graded:
        print(f"Experimental model: {hits}-{len(graded) - hits} ({hits / len(graded) * 100:.1f}%), "
              f"${profit:,.2f}, ROI {profit / (len(graded) * 100) * 100:.2f}%, {len(graded)} graded")
    else:
        print("Experimental model: no graded picks yet")

    if args.data_only:
        return
    if args.season != '2026-27':
        sys.exit("The HTML pages only track 2026-27; use --data-only for other seasons.")
    if len(form) != 32:
        missing = sorted(set(ABBR.values()) - set(form))
        sys.exit(f"No games yet for {', '.join(missing)} -- not updating index.html until every team has played.")
    update_index(os.path.join(root, 'index.html'), form, league, through)
    if odds is not None:
        update_experimental(os.path.join(root, 'experimental-model.html'), rows, through)
    print("Updated index.html" + (" and experimental-model.html" if odds is not None else ""))


if __name__ == '__main__':
    main()
