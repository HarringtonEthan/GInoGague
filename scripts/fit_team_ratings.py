"""
Fits attack/defense ratings for every team using the COMPLETE, finished
2025-26 season (all 1,312 games, final scores known). This is for
forward-looking prediction of 2026-27 games ONLY -- it is legitimate to use
a fully-completed season as training data for a genuinely future season.

This must NOT be used to re-score 2025-26 games themselves (that would be
fitting a model on the same games it's graded against). The 2025-26
Track Record stays the separate, honest, walk-forward backtest.

Model: for a game with home team h, away team a:
    E[home goals] = attack[h] * defense[a] * MU_HOME
    E[away goals] = attack[a] * defense[h] * MU_AWAY

attack[i] and defense[i] are normalized to mean 1 across the league.
MU_HOME / MU_AWAY are the league's actual average home/away goals per
game this season -- this is also how home-ice advantage enters the model
explicitly, which the previous (recency+shrinkage) model didn't have.

Fit via iterative proportional fitting (alternately rescale attack to match
each team's actual goals-for, then defense to match actual goals-against,
repeat to convergence) -- this satisfies the same equations as a Poisson
MLE fit for this multiplicative structure, no numpy/scipy required.
"""
import csv
import json

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


def load_games(path):
    games = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            games.append({'away': row[2], 'away_g': int(row[3]), 'home': row[4], 'home_g': int(row[5])})
    return games


def fit(games, teams, iterations=200):
    n = len(games)
    mu_home = sum(g['home_g'] for g in games) / n
    mu_away = sum(g['away_g'] for g in games) / n

    team_games = {t: [] for t in teams}
    for g in games:
        h, a = g['home'], g['away']
        team_games[h].append((a, True, g['home_g'], g['away_g']))
        team_games[a].append((h, False, g['away_g'], g['home_g']))

    attack = {t: 1.0 for t in teams}
    defense = {t: 1.0 for t in teams}

    for _ in range(iterations):
        new_attack = {}
        for t in teams:
            actual_gf = sum(gf for (_, _, gf, _) in team_games[t])
            denom = sum(defense[opp] * (mu_home if is_home else mu_away) for (opp, is_home, _, _) in team_games[t])
            new_attack[t] = actual_gf / denom if denom > 0 else 1.0
        mean_a = sum(new_attack.values()) / len(new_attack)
        attack = {t: v / mean_a for t, v in new_attack.items()}

        new_defense = {}
        for t in teams:
            actual_ga = sum(ga for (_, _, _, ga) in team_games[t])
            denom = sum(attack[opp] * (mu_away if is_home else mu_home) for (opp, is_home, _, _) in team_games[t])
            new_defense[t] = actual_ga / denom if denom > 0 else 1.0
        mean_b = sum(new_defense.values()) / len(new_defense)
        defense = {t: v / mean_b for t, v in new_defense.items()}

    return attack, defense, mu_home, mu_away


def main():
    games = load_games('games_2025-26.csv')
    teams = list(ABBR.keys())
    attack, defense, mu_home, mu_away = fit(games, teams)

    # verify convergence: check total actual vs total expected goals per team
    team_games = {t: [] for t in teams}
    for g in games:
        h, a = g['home'], g['away']
        team_games[h].append((a, True, g['home_g'], g['away_g']))
        team_games[a].append((h, False, g['away_g'], g['home_g']))

    max_err = 0
    for t in teams:
        actual_gf = sum(gf for (_, _, gf, _) in team_games[t])
        expected_gf = sum(attack[t] * defense[opp] * (mu_home if is_home else mu_away) for (opp, is_home, _, _) in team_games[t])
        err = abs(actual_gf - expected_gf)
        max_err = max(max_err, err)
    print(f"Convergence check: max |actual GF - expected GF| across all 32 teams = {max_err:.6f} goals (season total)")
    print(f"MU_HOME={mu_home:.6f}  MU_AWAY={mu_away:.6f}")

    out = {
        'mu_home': mu_home, 'mu_away': mu_away,
        'teams': {ABBR[t]: {'attack': attack[t], 'defense': defense[t]} for t in teams}
    }
    with open('team_ratings.json', 'w') as f:
        json.dump(out, f, indent=2)

    print("\nabbr  attack  defense")
    for t in sorted(teams, key=lambda x: -attack[x]):
        print(f"{ABBR[t]:5s} {attack[t]:.4f}  {defense[t]:.4f}")


if __name__ == '__main__':
    main()
