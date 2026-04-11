import os
from datetime import datetime, timezone
import json
import pandas as pd
import requests
from model import predict_score

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Missing BOT_TOKEN or CHAT_ID")

if not API_FOOTBALL_KEY:
    raise ValueError("Missing API_FOOTBALL_KEY")

SENT_FIXTURES_FILE = "sent_fixtures.json"


def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=data, timeout=30)
        if response.status_code != 200:
            print("Telegram ERROR:", response.text)
        else:
            print("Telegram sent successfully")
    except Exception as e:
        print("Telegram failed:", e)


def get_current_season():
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def normalize_team_name(name):
    mapping = {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Nottingham Forest": "Nott'm Forest",
        "Brighton & Hove Albion": "Brighton",
        "Wolverhampton Wanderers": "Wolves",
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Tottenham",
        "Leicester City": "Leicester",
        "Ipswich Town": "Ipswich",
    }
    return mapping.get(name, name)


def get_upcoming_fixtures():
    season = get_current_season()

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        "x-apisports-key": API_FOOTBALL_KEY
    }
    params = {
        "league": 39,   # Premier League
        "season": season,
        "next": 10
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    fixtures = []

    for item in data.get("response", []):
        home_raw = item["teams"]["home"]["name"]
        away_raw = item["teams"]["away"]["name"]
        home_team = normalize_team_name(home_raw)
        away_team = normalize_team_name(away_raw)
        fixture_date = item["fixture"]["date"]
        fixture_id = item["fixture"]["id"]

        fixtures.append({
            "fixture_id": fixture_id,
            "home_raw": home_raw,
            "away_raw": away_raw,
            "home_team": home_team,
            "away_team": away_team,
            "fixture_date": fixture_date
        })

    return fixtures


def parse_fixture_time(fixture_date_str):
    return datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))


def is_within_next_hours(fixture_date_str, hours=12):
    now = datetime.now(timezone.utc)
    kickoff = parse_fixture_time(fixture_date_str)
    diff = kickoff - now
    return 0 <= diff.total_seconds() <= hours * 3600


def load_sent_fixtures():
    if not os.path.exists(SENT_FIXTURES_FILE):
        return {}

    try:
        with open(SENT_FIXTURES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sent_fixtures(sent_data):
    with open(SENT_FIXTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(sent_data, f, indent=2)


def already_sent(sent_data, fixture_id):
    return str(fixture_id) in sent_data


def mark_as_sent(sent_data, fixture_id, payload):
    sent_data[str(fixture_id)] = payload


# Load historical CSV
df = pd.read_csv("data.csv")

teams = {}

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

    if hg > ag:
        teams[home]["form_points"].append(3)
        teams[away]["form_points"].append(0)
    elif hg < ag:
        teams[home]["form_points"].append(0)
        teams[away]["form_points"].append(3)
    else:
        teams[home]["form_points"].append(1)
        teams[away]["form_points"].append(1)

results = []
sent_data = load_sent_fixtures()

print("\n--- LIVE FIXTURE PICKS ---\n")

try:
    fixtures = get_upcoming_fixtures()
except Exception as e:
    error_message = f"API error: {str(e)}"
    print(error_message)
    send_telegram(error_message)
    raise

for fixture in fixtures:
    fixture_id = fixture["fixture_id"]
    raw_home = fixture["home_raw"]
    raw_away = fixture["away_raw"]
    home_team = fixture["home_team"]
    away_team = fixture["away_team"]
    fixture_date = fixture["fixture_date"]

    # Only check matches starting in next 12 hours
    if not is_within_next_hours(fixture_date, hours=12):
        continue

    # Prevent duplicate alerts
    if already_sent(sent_data, fixture_id):
        continue

    if home_team not in teams or away_team not in teams:
        continue

    if (
        len(teams[home_team]["home_scored"]) < 5
        or len(teams[away_team]["away_scored"]) < 5
        or len(teams[home_team]["form_points"]) < 5
        or len(teams[away_team]["form_points"]) < 5
    ):
        continue

    home_attack = sum(teams[home_team]["home_scored"][-5:]) / 5
    away_defence = sum(teams[away_team]["away_conceded"][-5:]) / 5

    away_attack = sum(teams[away_team]["away_scored"][-5:]) / 5
    home_defence = sum(teams[home_team]["home_conceded"][-5:]) / 5

    home_xg = ((home_attack + away_defence) / 2) * 1.10
    away_xg = ((away_attack + home_defence) / 2) * 0.90

    score, prob, _, home_win_prob, draw_prob, away_win_prob = predict_score(home_xg, away_xg)

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

    passes_filter = False

    if (
        main_market == "Home Win"
        and confidence >= 62
        and rating_gap >= 25
        and form_gap >= 2
    ):
        passes_filter = True

    if (
        main_market == "Away Win"
        and confidence >= 62
        and rating_gap >= 25
        and form_gap <= -2
    ):
        passes_filter = True

    if passes_filter:
        results.append((
            fixture_id,
            raw_home,
            raw_away,
            fixture_date,
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

results = sorted(results, key=lambda x: x[8], reverse=True)

if results:
    for (
        fixture_id,
        raw_home,
        raw_away,
        fixture_date,
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
            f"🚨 BET SIGNAL\n\n"
            f"{raw_home} vs {raw_away}\n"
            f"Kickoff: {fixture_date}\n\n"
            f"Main market: {main_market}\n"
            f"Confidence: {confidence}%\n"
            f"Safer market: {safer_market}\n\n"
            f"Predicted score: {score}\n"
            f"Correct score probability: {prob:.2%}\n"
            f"Home Win: {home_win_prob:.2%}\n"
            f"Draw: {draw_prob:.2%}\n"
            f"Away Win: {away_win_prob:.2%}\n"
            f"Rating gap: {rating_gap}%\n"
            f"Home form points (last 5): {home_form}\n"
            f"Away form points (last 5): {away_form}\n"
            f"Form gap: {form_gap}"
        )

        print(message)
        print()
        send_telegram(message)

        mark_as_sent(sent_data, fixture_id, {
            "home": raw_home,
            "away": raw_away,
            "date": fixture_date,
            "main_market": main_market,
            "confidence": confidence
        })

    save_sent_fixtures(sent_data)
else:
    print("No strong live-fixture picks found today.")
