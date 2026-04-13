import os
from datetime import datetime, timezone, timedelta
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

LEAGUES = [39, 140, 135, 78, 61]

LEAGUE_NAMES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

LOOKAHEAD_HOURS = 72
MAX_PICKS = 5

MIN_CONFIDENCE = 58
MIN_RATING_GAP = 20
MIN_HOME_FORM_GAP = 1
MIN_AWAY_FORM_GAP = -1


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


def normalize_team_name(name):
    mapping = {
        # England
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

        # Spain
        "Atletico Madrid": "Ath Madrid",
        "Espanyol": "Espanol",
        "Leganes": "Leganes",
        "Alaves": "Alaves",
        "Real Betis": "Betis",
        "Real Sociedad": "Sociedad",

        # Italy
        "AC Milan": "Milan",
        "AS Roma": "Roma",
        "Hellas Verona": "Verona",
        "Internazionale": "Inter",

        # Germany
        "Borussia Dortmund": "Dortmund",
        "Bayer Leverkusen": "Leverkusen",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "Borussia Monchengladbach": "M'gladbach",
        "FSV Mainz 05": "Mainz",
        "FC Augsburg": "Augsburg",
        "Union Berlin": "Union Berlin",
        "VfL Wolfsburg": "Wolfsburg",
        "1899 Hoffenheim": "Hoffenheim",

        # France
        "Paris Saint Germain": "Paris SG",
        "Olympique Marseille": "Marseille",
        "Olympique Lyonnais": "Lyon",
        "Stade Rennais FC": "Rennes",
        "LOSC Lille": "Lille",
        "OGC Nice": "Nice",
    }
    return mapping.get(name, name)


def get_upcoming_fixtures():
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}

    current_day = datetime.now(timezone.utc).date()
    all_fixtures = []
    seen_fixture_ids = set()

    for _ in range(3):  # today + next 2 days
        date_str = current_day.isoformat()

        for league_id in LEAGUES:
            params = {
                "league": league_id,
                "date": date_str
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            count = len(data.get("response", []))
            print(f"{date_str} | League {league_id} fixtures: {count}")

            for item in data.get("response", []):
                fixture_id = item["fixture"]["id"]

                if fixture_id in seen_fixture_ids:
                    continue

                seen_fixture_ids.add(fixture_id)

                home_raw = item["teams"]["home"]["name"]
                away_raw = item["teams"]["away"]["name"]

                all_fixtures.append({
                    "fixture_id": fixture_id,
                    "league_id": league_id,
                    "league_name": LEAGUE_NAMES.get(league_id, "Unknown League"),
                    "home_raw": home_raw,
                    "away_raw": away_raw,
                    "home_team": normalize_team_name(home_raw),
                    "away_team": normalize_team_name(away_raw),
                    "fixture_date": item["fixture"]["date"]
                })

        current_day = current_day + timedelta(days=1)

    return all_fixtures


def parse_fixture_time(fixture_date_str):
    return datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))


def is_within_next_hours(fixture_date_str, hours=72):
    now = datetime.now(timezone.utc)
    kickoff = parse_fixture_time(fixture_date_str)
    diff = kickoff - now
    return 0 <= diff.total_seconds() <= hours * 3600


# Load historical CSV
df = pd.read_csv("data.csv")
played_matches = df[df["FTHG"].notna() & df["FTAG"].notna()]

teams = {}

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

print("\n--- LIVE FIXTURE PICKS ---\n")

skip_counts = {
    "outside_window": 0,
    "team_not_found": 0,
    "not_enough_data": 0,
    "filter_failed": 0,
    "passed": 0,
}

results = []

try:
    fixtures = get_upcoming_fixtures()
    print(f"Fetched fixtures: {len(fixtures)}")
except Exception as e:
    error_message = f"API error: {str(e)}"
    print(error_message)
    send_telegram(error_message)
    raise

for fixture in fixtures:
    fixture_id = fixture["fixture_id"]
    league_name = fixture["league_name"]
    raw_home = fixture["home_raw"]
    raw_away = fixture["away_raw"]
    home_team = fixture["home_team"]
    away_team = fixture["away_team"]
    fixture_date = fixture["fixture_date"]

    print(f"Checking: {league_name} | {raw_home} vs {raw_away}")

    if not is_within_next_hours(fixture_date, hours=LOOKAHEAD_HOURS):
        skip_counts["outside_window"] += 1
        continue

    if home_team not in teams or away_team not in teams:
        skip_counts["team_not_found"] += 1
        continue

    if (
        len(teams[home_team]["home_scored"]) < 5
        or len(teams[away_team]["away_scored"]) < 5
        or len(teams[home_team]["form_points"]) < 5
        or len(teams[away_team]["form_points"]) < 5
    ):
        skip_counts["not_enough_data"] += 1
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
        and confidence >= MIN_CONFIDENCE
        and rating_gap >= MIN_RATING_GAP
        and form_gap >= MIN_HOME_FORM_GAP
    ):
        passes_filter = True

    if (
        main_market == "Away Win"
        and confidence >= MIN_CONFIDENCE
        and rating_gap >= MIN_RATING_GAP
        and form_gap <= MIN_AWAY_FORM_GAP
    ):
        passes_filter = True

    if not passes_filter:
        skip_counts["filter_failed"] += 1
        continue

    skip_counts["passed"] += 1

    results.append((
        fixture_id,
        league_name,
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

results = sorted(results, key=lambda x: x[9], reverse=True)

print("\n--- DEBUG SUMMARY ---")
for key, value in skip_counts.items():
    print(f"{key}: {value}")

if results:
    for (
        fixture_id,
        league_name,
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
    ) in results[:MAX_PICKS]:

        message = (
            f"🚨 BET SIGNAL\n\n"
            f"🏆 {league_name}\n"
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

        print("\n" + message + "\n")
        send_telegram(message)
else:
    print("No strong live-fixture picks found today.")
