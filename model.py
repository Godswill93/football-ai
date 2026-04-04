from scipy.stats import poisson


def predict_score(home_xg, away_xg):
    max_goals = 6
    probabilities = {}

    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
            probabilities[(home_goals, away_goals)] = prob

    best_score = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_score]

    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0

    for (home_goals, away_goals), prob in probabilities.items():
        if home_goals > away_goals:
            home_win_prob += prob
        elif home_goals == away_goals:
            draw_prob += prob
        else:
            away_win_prob += prob

    best_score_str = f"{best_score[0]}-{best_score[1]}"

    return best_score_str, best_prob, probabilities, home_win_prob, draw_prob, away_win_prob