import os
import pandas as pd
import requests
from model import predict_score

BOT_TOKEN = "8516344376:AAE-h-C3Y2Ba2OPRCb8S3Gk6CPkWEzNonHM"
CHAT_ID = "6110133218"


def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=data)


# Load CSV
df = pd.read_csv("data.csv")

teams = {}

# Use only played matches
played_matches = df[df["FTHG"].notna() & df["FTAG"].notna()]

for _, row in played_matches.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    hg = row["FTHG"]
    ag = row["FTAG"]

    if home not in teams:
        teams[home] = {
            "home_scored": [],
            "home_conceded": [],
            "away_scored": [],
            "away_conceded": [],
            "form_points": []
        }

    if away not in teams:
        teams[away] = {
            "home_scored": [],
            "home_conceded": [],
            "away_scored": [],
            "away_conceded": [],
            "form_points": []
        }

    teams[home]["home_scored"].append(hg)
    teams[home]["home_conceded"].append(ag)

    teams[away]["away_scored"].append(ag)
    teams[away]["away_conceded"].append(hg)

    # Form points
    if hg > ag:
        teams[home]["form_points"].append(3)
        teams[away]["form_points"].append(0)
    elif hg < ag:
        teams[home]["form_points"].append(0)
        teams[away]["form_points"].append(3)
    else:
        teams[home]["form_points"].append(1)
        teams[away]["form_points"].append(1)

fixtures = [
    ("Liverpool", "Everton"),
    ("Arsenal", "Crystal Palace"),
    ("Chelsea", "Bournemouth"),
    ("Man City", "Aston Villa"),
    ("Tottenham", "Nott'm Forest")
]

results = []

print("\n--- FILTERED PICKS (FORM + RATING GAP VERSION) ---\n")

for home_team, away_team in fixtures:
    if home_team not in teams or away_team not in teams:
        continue

    if (
        len(teams[home_team]["home_scored"]) < 5
        or len(teams[away_team]["away_scored"]) < 5
        or len(teams[home_team]["form_points"]) < 5
        or len(teams[away_team]["form_points"]) < 5
    ):
        continue

    # Last 5 home/away form
    home_attack = sum(teams[home_team]["home_scored"][-5:]) / 5
    away_defence = sum(teams[away_team]["away_conceded"][-5:]) / 5

    away_attack = sum(teams[away_team]["away_scored"][-5:]) / 5
    home_defence = sum(teams[home_team]["home_conceded"][-5:]) / 5

    # Home advantage
    home_xg = ((home_attack + away_defence) / 2) * 1.10
    away_xg = ((away_attack + home_defence) / 2) * 0.90

    score, prob, _, home_win_prob, draw_prob, away_win_prob = predict_score(home_xg, away_xg)

    # Last 5 form points
    home_form = sum(teams[home_team]["form_points"][-5:])
    away_form = sum(teams[away_team]["form_points"][-5:])
    form_gap = home_form - away_form

    if home_win_prob > draw_prob and home_win_prob > away_win_prob:
        main_market = "Home Win"
        confidence = round(home_win_prob * 100, 1)
    elif away_win_prob > draw_prob and away_win_prob > home_win_prob:
        main_market = "Away Win"
        confidence = round(away_win_prob * 100, 1)
    else:
        main_market = "Draw"
        confidence = round(draw_prob * 100, 1)

    home_goals, away_goals = map(int, score.split("-"))
    total_goals = home_goals + away_goals

    if home_goals > 0 and away_goals > 0:
        safer_market = "BTTS"
    elif total_goals >= 3:
        safer_market = "Over 2.5 Goals"
    else:
        safer_market = "Under 3.5 Goals"

    rating_gap = abs(home_win_prob - away_win_prob) * 100

    # Stronger filter
    passes_filter = False

    if (
        main_market == "Home Win"
        and confidence >= 60
        and rating_gap >= 25
        and form_gap >= 3
    ):
        passes_filter = True

    if (
        main_market == "Away Win"
        and confidence >= 60
        and rating_gap >= 25
        and form_gap <= -3
    ):
        passes_filter = True

    if passes_filter:
        results.append((
            home_team,
            away_team,
            score,
            prob,
            safer_market,
            main_market,
            confidence,
            home_win_prob,
            draw_prob,
            away_win_prob,
            round(rating_gap, 1),
            home_form,
            away_form,
            form_gap
        ))

results = sorted(results, key=lambda x: x[6], reverse=True)

if not results:
    message = "No strong picks found today."
    print(message)
    send_telegram(message)
else:
    for (
        home_team,
        away_team,
        score,
        prob,
        safer_market,
        main_market,
        confidence,
        home_win_prob,
        draw_prob,
        away_win_prob,
        rating_gap,
        home_form,
        away_form,
        form_gap
    ) in results[:2]:

        message = (
            f"{home_team} vs {away_team}\n\n"
            f"Predicted score: {score}\n"
            f"Correct score probability: {prob:.2%}\n"
            f"Main market: {main_market}\n"
            f"Confidence: {confidence}%\n"
            f"Home Win: {home_win_prob:.2%}\n"
            f"Draw: {draw_prob:.2%}\n"
            f"Away Win: {away_win_prob:.2%}\n"
            f"Rating gap: {rating_gap}%\n"
            f"Home form points (last 5): {home_form}\n"
            f"Away form points (last 5): {away_form}\n"
            f"Form gap: {form_gap}\n"
            f"Safer market: {safer_market}"
        )

        print(message)
        print()
        send_telegram(message)
