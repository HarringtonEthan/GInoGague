"""
Build this season's data files from a checkout of the public
github.com/Prezbo8/nhl-total-model repo, which republishes NHL results and
sportsbook closing totals every day:

    git clone --depth 1 https://github.com/Prezbo8/nhl-total-model ../nhl-total-model
    python3 scripts/import_live_data.py ../nhl-total-model
    python3 scripts/update_live_season.py

Writes, in the same formats as the 2025-26 files:
  data_games_2026-27.csv  every completed game (date, ET start, teams, final score)
  data_odds_2026-27.csv   closing total + over/under odds for every game that has one

What comes from where:
  - data/paper_trades.csv: date, teams, final score and closing total/odds
    for every game from Oct 1 on.
  - That file starts Oct 1, so the season's first 8 games (Sep 29-30) are
    listed below by hand from published recaps (ESPN, NHL.com). They are each
    team's first game, so the model never makes a pick on them -- they only
    feed teams' scoring logs -- and no closing line is needed for them.
  - data/game_starts.csv + data/goalie_*.csv: each game's NHL game ID and
    start time, and the goals each goalie allowed. Every final score is
    checked against those: a team can't have scored fewer goals than the
    other side's goalies allowed (it can score more: empty-net goals and
    shootout winners aren't charged to a goalie).
This script only reads that checkout's CSV files; it never runs its code.
"""
import csv
import datetime
import os
import re
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON_PREFIX = '202602'  # NHL game IDs for 2026-27 regular-season games
EASTERN = ZoneInfo('America/New_York')

# (date, away, home, away goals, home goals, OT/SO)
OPENING_GAMES = [
    ('2026-09-29', 'FLA', 'CAR', 1, 0, 'OT'),
    ('2026-09-29', 'MTL', 'TOR', 3, 2, ''),
    ('2026-09-29', 'NYR', 'BOS', 0, 3, ''),
    ('2026-09-29', 'VAN', 'EDM', 6, 5, 'OT'),
    ('2026-09-29', 'CHI', 'VGK', 2, 5, ''),
    ('2026-09-30', 'PIT', 'PHI', 7, 0, ''),
    ('2026-09-30', 'LAK', 'COL', 4, 8, ''),
    ('2026-09-30', 'NYI', 'TOR', 1, 2, ''),
]


def team_names():
    """abbr -> full name, from update_live_season.py's table (current names only)."""
    src = open(os.path.join(ROOT, 'scripts', 'update_live_season.py')).read()
    pairs = re.findall(r"'([^']+)':'([A-Z]{3})'", re.search(r'ABBR = \{.*?\n\}', src, re.S).group(0))
    names = {}
    for full, abbr in pairs:
        if full != 'Utah Hockey Club':
            names[abbr] = full
    return names


def num(s):
    return int(float(s))


def fmt_odds(o):
    return f'+{o}' if o > 0 else str(o)


def fmt_total(t):
    t = float(t)
    return str(int(t)) if t == int(t) else str(t)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data = os.path.join(sys.argv[1], 'data')
    names = team_names()

    # NHL game ID for each (date, {team, team}), plus start times and goalie goals allowed
    teams_by_id = defaultdict(set)
    team_of_goalie = {}
    for r in csv.DictReader(open(os.path.join(data, 'goalie_starts.csv'))):
        if r['gameId'].startswith(SEASON_PREFIX):
            teams_by_id[r['gameId']].add(r['team'])
            team_of_goalie[(r['gameId'], r['playerId'])] = r['team']
    date_by_id = {}
    allowed = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(os.path.join(data, 'goalie_games.csv'))):
        gid = r['gameId'].split('.')[0]
        if not gid.startswith(SEASON_PREFIX):
            continue
        d = r['gameDate'].split('.')[0]
        date_by_id[gid] = f'{d[:4]}-{d[4:6]}-{d[6:]}'
        allowed[gid][team_of_goalie.get((gid, r['playerId']))] += num(r['goals'])
    start_by_id = {r['gameId']: r['start_utc'] for r in csv.DictReader(open(os.path.join(data, 'game_starts.csv')))}
    id_by_game = {(date_by_id[g], frozenset(t)): g for g, t in teams_by_id.items() if g in date_by_id}

    games = [{'date': d, 'away': a, 'home': h, 'away_g': ag, 'home_g': hg, 'ot': ot, 'line': None}
             for d, a, h, ag, hg, ot in OPENING_GAMES]
    pending = 0
    for r in csv.DictReader(open(os.path.join(data, 'paper_trades.csv'))):
        if r['away_score'] == '' or r['home_score'] == '':
            pending += 1
            continue
        line = None
        if r['close_total'] and r['close_over'] and r['close_under']:
            line = (r['close_total'], num(r['close_over']), num(r['close_under']))
        games.append({'date': r['date'], 'away': r['away'], 'home': r['home'],
                      'away_g': num(r['away_score']), 'home_g': num(r['home_score']), 'ot': '', 'line': line})

    problems = []
    seen = set()
    for g in games:
        for t in (g['away'], g['home']):
            if t not in names:
                problems.append(f"{g['date']}: unknown team {t}")
        key = (g['date'], frozenset((g['away'], g['home'])))
        if key in seen:
            problems.append(f"{g['date']} {g['away']}@{g['home']}: listed twice")
        seen.add(key)
        gid = id_by_game.get(key)
        g['start'] = ''
        if gid is None:
            problems.append(f"{g['date']} {g['away']}@{g['home']}: no NHL game ID found to check it against")
            continue
        utc = datetime.datetime.fromisoformat(start_by_id[gid].replace('Z', '+00:00'))
        g['start'] = utc.astimezone(EASTERN).strftime('%-I:%M %p')
        if g['away_g'] < allowed[gid][g['home']] or g['home_g'] < allowed[gid][g['away']]:
            problems.append(f"{g['date']} {g['away']}@{g['home']}: final {g['away_g']}-{g['home_g']} is below "
                            f"the goals the goalies allowed ({allowed[gid][g['home']]}-{allowed[gid][g['away']]})")
    # every season game the NHL data says has been played must be here
    have = {id_by_game.get((g['date'], frozenset((g['away'], g['home'])))) for g in games}
    for gid in sorted(set(date_by_id) - have):
        problems.append(f"{date_by_id[gid]} {'/'.join(sorted(teams_by_id[gid]))} (game {gid}): played, but no final score")
    if problems:
        sys.exit('Not writing anything:\n  ' + '\n  '.join(problems))

    def sort_key(g):
        t = datetime.datetime.strptime(g['start'], '%I:%M %p').time() if g['start'] else datetime.time(0)
        return (g['date'], t, g['away'])
    games.sort(key=sort_key)

    with open(os.path.join(ROOT, 'data_games_2026-27.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Date', 'Time', 'Visitor', 'G', 'Home', 'G', '', 'Att.', 'LOG', 'Notes'])
        for g in games:
            w.writerow([g['date'], g['start'], names[g['away']], g['away_g'], names[g['home']], g['home_g'],
                        g['ot'], '', '', ''])
    with_lines = [g for g in games if g['line']]
    with open(os.path.join(ROOT, 'data_odds_2026-27.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Date', 'Start Time (Sask)', 'Start Time (ET)', 'Visitor', 'Score', 'Home', 'Score', 'Status',
                    'Visitor Goalie', 'Home Goalie', 'O/U', 'Over', 'Under', 'Away ML', 'Home ML',
                    'Home PL Spread', 'PL Away', 'PL Home'])
        for g in with_lines:
            total, over, under = g['line']
            w.writerow([g['date'], '', g['start'], names[g['away']], g['away_g'], names[g['home']], g['home_g'], '',
                        '', '', fmt_total(total), fmt_odds(over), fmt_odds(under), '', '', '', '', ''])

    print(f"{len(games)} completed games ({games[0]['date']} to {games[-1]['date']}), "
          f"{len(with_lines)} with closing lines; {pending} listed game(s) not final yet in the source.")


if __name__ == '__main__':
    main()
