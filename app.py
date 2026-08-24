import streamlit as st
import pandas as pd
from features import create_features, get_latest_team_stats, get_live_team_stats
from model import split_data, train_model
 
st.set_page_config(page_title="Premier League Predictor", layout="wide")
 
 
@st.cache_data
def load_data():
    return pd.read_csv("2025_2026 dataset.csv")

@st.cache_data
def load_previous_season():
    return pd.read_csv("E0 (1).csv")
 
 
@st.cache_resource
def train():
    data = load_data()
    features = create_features(data)
    x = features[[
        "AttackDifference",
        "DefenseDifference",
        "FormDifference",
        "FormDefenseDifference",
        "PointsDifference",
    ]]
    y = features["Outcome"]
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = train_model(x_train, y_train)
    return model
 
 
data = load_data()
model = train()
previous_season = load_previous_season()
latest_stats = get_live_team_stats(data, previous_season, min_matches=5)
 
st.title("⚽ Premier League Match Predictor")
 
tab1, tab2 = st.tabs(["Predict a Match", "League Table"])
 
with tab1:
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

        if home["IsEstimated"] or away["IsEstimated"]:
            st.info(
                f"⚠️ Limited current-season data for "
                f"{home_team if home['IsEstimated'] else ''} "
                f"{away_team if away['IsEstimated'] else ''} — "
                f"stats blended with {home['BaselineSource'] if home['IsEstimated'] else away['BaselineSource']} data."
            )
 
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
 
        outcome_labels = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}
 
        st.subheader(f"Prediction: {outcome_labels[pred]}")
 
        proba_pairs = sorted(
            zip(classes, proba), key=lambda p: p[1], reverse=True
        )

        for outcome_code, confidence in proba_pairs:
            label = outcome_labels[outcome_code]
            st.write(f"**{label}** — {confidence:.1%}")
            st.progress(confidence)
 
        st.markdown("#### What favored each team")
        compare_df = pd.DataFrame({
            home_team: [
                home["AvgGoalsScored"], home["AvgGoalsConceded"],
                home["Last5Goals"], home["Last5Conceded"],
                home["WinRate"], home["PointsTotal"],
            ],
            away_team: [
                away["AvgGoalsScored"], away["AvgGoalsConceded"],
                away["Last5Goals"], away["Last5Conceded"],
                away["WinRate"], away["PointsTotal"],
            ],
        }, index=[
            "Avg Goals Scored", "Avg Goals Conceded",
            "Last 5 Goals", "Last 5 Conceded",
            "Win Rate", "Points",
        ])
        st.dataframe(compare_df, use_container_width=True)
 
with tab2:
    st.subheader("League Table (current standings)")
    table = latest_stats.copy()
    table["Points"] = table["PointsTotal"]
    table = table.sort_values("Points", ascending=False)
    table = table[["MatchesPlayed", "Points", "WinRate", "AvgGoalsScored", "AvgGoalsConceded"]]
    table.index.name = "Team"
    st.dataframe(table, use_container_width=True)
 
