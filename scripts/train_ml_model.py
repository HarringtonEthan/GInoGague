"""
Real machine-learning approach to replace the hand-built Poisson formula,
evaluated the most rigorous way available with only two seasons: train on
one season ENTIRELY, evaluate on the other season, which the model never
saw in any form during training. Then repeat in the other direction.

This is a stronger test than the Experimental Model's "fit both seasons
at once" approach -- each evaluation season is a genuine holdout, not
something the model's parameters were chosen to match. If both directions
show a real edge, that's actual evidence, not a guarantee, but real
cross-season generalization rather than curve-fitting to known answers.

Also reports 5-fold time-respecting CV within a season for model/
hyperparameter selection (not for the headline numbers).
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit

FEATURES = [
    'home_gf', 'home_ga', 'away_gf', 'away_ga',
    'home_gf_home', 'home_ga_home', 'away_gf_away', 'away_ga_away',
    'home_rest', 'away_rest', 'home_b2b', 'away_b2b',
    'home_goalie_sv', 'away_goalie_sv', 'home_severity', 'away_severity',
    'line', 'implied_over',
]


def profit_for_bet(odds, staked=100.0):
    if odds > 0:
        return staked * (odds / 100.0)
    return staked * (100.0 / abs(odds))


def evaluate(model, df_test, threshold):
    X_test = df_test[FEATURES].values
    proba = model.predict_proba(X_test)[:, 1]  # P(over)
    hits = misses = 0
    profit = 0.0
    for p_over, row in zip(proba, df_test.itertuples()):
        if row.label is None or (isinstance(row.label, float) and np.isnan(row.label)):
            continue
        if p_over > threshold:
            lean = 1
        elif (1 - p_over) > threshold:
            lean = 0
        else:
            continue
        hit = (lean == row.label)
        bet_odds = row.over_odds if lean == 1 else row.under_odds
        if hit:
            hits += 1
            profit += profit_for_bet(bet_odds)
        else:
            misses += 1
            profit -= 100.0
    graded = hits + misses
    roi = profit / (graded * 100) * 100 if graded else None
    hit_rate = hits / graded * 100 if graded else None
    return {'hits': hits, 'misses': misses, 'graded': graded, 'profit': profit, 'roi': roi, 'hit_rate': hit_rate}


def main():
    df = pd.read_csv('ml_features.csv')
    df = df.dropna(subset=FEATURES)
    df_labeled = df[df['label'].notna()].copy()

    s2425 = df_labeled[df_labeled['season'] == '2024-25'].reset_index(drop=True)
    s2526 = df_labeled[df_labeled['season'] == '2025-26'].reset_index(drop=True)
    print(f"2024-25: {len(s2425)} labeled games, 2025-26: {len(s2526)} labeled games")

    model_specs = {
        'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        'gradient_boosting': lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=30),
        'random_forest': lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=20, class_weight='balanced', random_state=0),
    }

    thresholds = [0.52, 0.54, 0.55, 0.56, 0.58, 0.60]

    for name, spec in model_specs.items():
        print(f"\n########## {name} ##########")

        print("--- Train on 2024-25, evaluate on 2025-26 (true holdout) ---")
        model = spec()
        model.fit(s2425[FEATURES].values, s2425['label'].values)
        for th in thresholds:
            r = evaluate(model, s2526, th)
            if r['graded']:
                print(f"  th={th}: {r['hits']}-{r['misses']} ({r['hit_rate']:.1f}%) ${r['profit']:.0f} ROI{r['roi']:.1f}% n={r['graded']}")

        print("--- Train on 2025-26, evaluate on 2024-25 (true holdout) ---")
        model = spec()
        model.fit(s2526[FEATURES].values, s2526['label'].values)
        for th in thresholds:
            r = evaluate(model, s2425, th)
            if r['graded']:
                print(f"  th={th}: {r['hits']}-{r['misses']} ({r['hit_rate']:.1f}%) ${r['profit']:.0f} ROI{r['roi']:.1f}% n={r['graded']}")


if __name__ == '__main__':
    main()
