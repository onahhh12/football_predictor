import streamlit as st
import pandas as pd
import requests
import tomllib
from features import create_features, get_live_team_stats
from model import split_data, train_model

st.set_page_config(page_title="Premier League Predictor", layout="wide")

# --- Secrets (loaded once, at the top, so every function below can see them) ---
with open("resources/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

FOOTBALL_DATA_KEY = secrets["API_KEY"]

CURRENT_SEASON_TEAMS = [
    "Liverpool", "Arsenal", "Man City", "Chelsea", "Newcastle",
    "Aston Villa", "Nott'm Forest", "Brighton", "Bournemouth", "Fulham",
    "Brentford", "Crystal Palace", "Everton", "Hull City", "Man Utd",
    "Coventry City", "Spurs", "Leeds", "Ipswich Town", "Sunderland",
]


@st.cache_data
def load_data():
    return pd.read_csv("2026_2027 dataset.csv")


@st.cache_data
def load_previous_season():
    return pd.read_csv("2025_2026 dataset.csv")


@st.cache_resource
def train():
    """
    Returns a trained model, or None only in the unlikely case that even
    combined data isn't enough to stratify-split (e.g. missing files).
    Training data = last season's full results + this season's matches so far.
    """
    current = load_data()
    previous = load_previous_season()

    combined = pd.concat([previous, current], ignore_index=True)

    if len(combined) < 10:
        return None

    features = create_features(combined)

    if features["Outcome"].nunique() < 3:
        return None

    x = features[[
        "AttackDifference",
        "DefenseDifference",
        "FormDifference",
        "FormDefenseDifference",
        "PointsDifference",
    ]]
    y = features["Outcome"]

    try:
        x_train, x_test, y_train, y_test = split_data(x, y)
        model = train_model(x_train, y_train)
        return model
    except ValueError:
        return None


@st.cache_data(ttl=3600)
def get_upcoming_fixtures():
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    params = {"status": "SCHEDULED"}

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    matches = data["matches"]
    if not matches:
        return pd.DataFrame(columns=["Date", "Home", "Away"])

    next_matchday = min(m["matchday"] for m in matches)
    upcoming = [m for m in matches if m["matchday"] == next_matchday]

    rows = []
    for m in upcoming:
        kickoff = pd.to_datetime(m["utcDate"])
        rows.append({
            "Date": kickoff.strftime("%a %d %b, %H:%M"),
            "Home": m["homeTeam"]["name"],
            "Away": m["awayTeam"]["name"],
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def get_live_standings():
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    table_entries = data["standings"][0]["table"]

    rows = []
    for entry in table_entries:
        rows.append({
            "Team": entry["team"]["name"],
            "Played": entry["playedGames"],
            "Points": entry["points"],
            "Won": entry["won"],
            "Drawn": entry["draw"],
            "Lost": entry["lost"],
            "GoalsFor": entry["goalsFor"],
            "GoalsAgainst": entry["goalsAgainst"],
            "GoalDiff": entry["goalDifference"],
        })

    table = pd.DataFrame(rows).set_index("Team")
    return table


def heuristic_prediction(home, away):
    """
    Simple fallback used when there's no trained model yet.
    """
    home_score = (
        home["AvgGoalsScored"] - home["AvgGoalsConceded"] + home["PointsTotal"] / 10
    )
    away_score = (
        away["AvgGoalsScored"] - away["AvgGoalsConceded"] + away["PointsTotal"] / 10
    )

    diff = home_score - away_score

    if diff > 0.3:
        return "H", diff
    elif diff < -0.3:
        return "A", diff
    else:
        return "D", diff


data = load_data()
model = train()
previous_season = load_previous_season()
latest_stats = get_live_team_stats(data, previous_season, CURRENT_SEASON_TEAMS, min_matches=5)

st.title("⚽ Premier League Match Predictor")

tab1, tab2 = st.tabs(["Predict a Match", "League Table"])

with tab1:
    st.markdown("#### Upcoming Fixtures")
    try:
        fixtures = get_upcoming_fixtures()
        if fixtures.empty:
            st.caption("No upcoming fixtures found.")
        else:
            st.dataframe(fixtures, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Couldn't fetch fixtures: {e}")

    st.markdown("---")

    teams = sorted(latest_stats.index.tolist())
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=1)

    if home_team == away_team:
        st.warning("Pick two different teams.")
    else:
        home = latest_stats.loc[home_team]
        away = latest_stats.loc[away_team]

        outcome_labels = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}

        if model is not None:
            x_new = pd.DataFrame([{
                "AttackDifference": home["AvgGoalsScored"] - away["AvgGoalsScored"],
                "DefenseDifference": away["AvgGoalsConceded"] - home["AvgGoalsConceded"],
                "FormDifference": home["Last5Goals"] - away["Last5Goals"],
                "FormDefenseDifference": away["Last5Conceded"] - home["Last5Conceded"],
                "PointsDifference": home["PointsTotal"] - away["PointsTotal"],
            }])

            proba = model.predict_proba(x_new)[0]
            classes = model.classes_
            pred = model.predict(x_new)[0]

            st.subheader(f"Prediction: {outcome_labels[pred]}")

            proba_pairs = sorted(
                zip(classes, proba), key=lambda p: p[1], reverse=True
            )

            for outcome_code, confidence in proba_pairs:
                label = outcome_labels[outcome_code]
                st.write(f"**{label}** — {confidence:.1%}")
                st.progress(confidence)
        else:
            pred, lean = heuristic_prediction(home, away)
            st.subheader(f"Estimated lean: {outcome_labels[pred]}")
            st.caption(
                "This is a rough estimate based on goal and points differences, "
                "not a trained model prediction."
            )

with tab2:
    st.subheader("League Table (live standings)")
    try:
        table = get_live_standings()
        st.dataframe(table, use_container_width=True)
    except Exception as e:
        st.error(f"Couldn't fetch live standings: {e}")