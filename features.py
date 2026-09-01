import pandas as pd

def create_features(data):
    data = data.copy()

    goals = data["Result"].str.split("-", expand=True)
    goals = goals.astype(int)
    goals.columns = ["HomeGoals", "AwayGoals"]

    data["HomeGoals"] = goals["HomeGoals"]
    data["AwayGoals"] = goals["AwayGoals"]
    data["TotalGoals"] = data["HomeGoals"] + data["AwayGoals"]

    data["Outcome"] = "D"
    data.loc[data["HomeGoals"] > data["AwayGoals"], "Outcome"] = "H"
    data.loc[data["HomeGoals"] < data["AwayGoals"], "Outcome"] = "A"

    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
        
    home = data[["Date", "Home Team", "Away Team", "HomeGoals", "AwayGoals", "Outcome"]].copy()
    home["Team"] = home["Home Team"]
    home["Opponent"] = home["Away Team"]
    home["GoalsScored"] = home["HomeGoals"]
    home["GoalsConceded"] = home["AwayGoals"]

    away = data[["Date", "Home Team", "Away Team", "HomeGoals", "AwayGoals", "Outcome"]].copy()
    away["Team"] = away["Away Team"]
    away["Opponent"] = away["Home Team"]
    away["GoalsScored"] = away["AwayGoals"]
    away["GoalsConceded"] = away["HomeGoals"]

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values("Date").reset_index(drop=True)

    team_matches["GoalsBefore"] = (
        team_matches.groupby("Team")["GoalsScored"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )

    team_matches["GoalsConcededBefore"] = (
        team_matches.groupby("Team")["GoalsConceded"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )

    team_matches["MatchesPlayed"] = team_matches.groupby("Team").cumcount()

    team_matches["AvgGoalsScored"] = (
        team_matches["GoalsBefore"] / team_matches["MatchesPlayed"]
    ).fillna(0)

    team_matches["AvgGoalsConceded"] = (
        team_matches["GoalsConcededBefore"] / team_matches["MatchesPlayed"]
    ).fillna(0)

    team_matches["Last5Goals"] = (
        team_matches.groupby("Team")["GoalsScored"]
        .transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        .fillna(0)
    )
    team_matches["Last5Conceded"] = (
        team_matches.groupby("Team")["GoalsConceded"]
        .transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        .fillna(0)
    )

    team_matches["Win"] = (
        team_matches["GoalsScored"] > team_matches["GoalsConceded"]
    ).astype(int)
 
    team_matches["WinsBefore"] = (
        team_matches.groupby("Team")["Win"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
 
    team_matches["WinRateBefore"] = (
        team_matches["WinsBefore"] / team_matches["MatchesPlayed"].replace(0, float("nan"))
    ).fillna(0)
 
    #Home/away-specific win rates 
    team_matches["IsHome"] = (team_matches["Team"] == team_matches["Home Team"]).astype(int)
    team_matches["IsAway"] = (team_matches["Team"] == team_matches["Away Team"]).astype(int)
    team_matches["HomeWin"] = team_matches["Win"] * team_matches["IsHome"]
    team_matches["AwayWin"] = team_matches["Win"] * team_matches["IsAway"]
 
    team_matches["HomeWinsBefore"] = (
        team_matches.groupby("Team")["HomeWin"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    team_matches["AwayWinsBefore"] = (
        team_matches.groupby("Team")["AwayWin"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    team_matches["HomeMatchesPlayed"] = (
        team_matches.groupby("Team")["IsHome"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    team_matches["AwayMatchesPlayed"] = (
        team_matches.groupby("Team")["IsAway"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
 
    team_matches["HomeWinRate"] = (
        team_matches["HomeWinsBefore"] / team_matches["HomeMatchesPlayed"].replace(0, float("nan"))
    ).fillna(0)
    team_matches["AwayWinRate"] = (
        team_matches["AwayWinsBefore"] / team_matches["AwayMatchesPlayed"].replace(0, float("nan"))
    ).fillna(0)

    team_matches["Points"] = 0
    team_matches.loc[team_matches["Win"] == 1, "Points"] = 3
    team_matches.loc[team_matches["GoalsScored"] == team_matches["GoalsConceded"], "Points"] = 1

    team_matches["PointsBefore"] = (
        team_matches.groupby("Team")["Points"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )

    #Split back into home-side / away-side feature rows
    home_features = team_matches[team_matches["Team"] == team_matches["Home Team"]].copy()
    home_features = home_features.rename(columns={
        "AvgGoalsScored": "HomeAvgGoalsBefore",
        "AvgGoalsConceded": "HomeAvgConcededBefore",
        "Last5Goals": "HomeLast5Goals",
        "Last5Conceded": "HomeLast5Conceded",
        "WinRateBefore": "HomeWinRateBefore",
        "PointsBefore": "HomePointsBefore",
    })
 
    away_features = team_matches[team_matches["Team"] == team_matches["Away Team"]].copy()
    away_features = away_features.rename(columns={
        "AvgGoalsScored": "AwayAvgGoalsBefore",
        "AvgGoalsConceded": "AwayAvgConcededBefore",
        "Last5Goals": "AwayLast5Goals",
        "Last5Conceded": "AwayLast5Conceded",
        "WinRateBefore": "AwayWinRateBefore",
        "PointsBefore": "AwayPointsBefore",
    })
 
    match_data = home_features.merge(
        away_features,
        on=["Date", "Home Team", "Away Team"],
        how="left",
    )
 
    #Final feature set 
    features = match_data[[
        "Date",
        "Home Team",
        "Away Team",
        "HomeAvgGoalsBefore",
        "HomeAvgConcededBefore",
        "HomeLast5Goals",
        "HomeLast5Conceded",
        "HomeWinRateBefore",
        "HomePointsBefore",
        "AwayAvgGoalsBefore",
        "AwayAvgConcededBefore",
        "AwayLast5Goals",
        "AwayLast5Conceded",
        "AwayWinRateBefore",
        "AwayPointsBefore",
    ]].copy()
 
    features["AttackDifference"] = (
        features["HomeAvgGoalsBefore"] - features["AwayAvgGoalsBefore"]
    )
    features["DefenseDifference"] = (
        features["AwayAvgConcededBefore"] - features["HomeAvgConcededBefore"]
    )
    features["FormDifference"] = (
        features["HomeLast5Goals"] - features["AwayLast5Goals"]
    )
    features["FormDefenseDifference"] = (
        features["AwayLast5Conceded"] - features["HomeLast5Conceded"]
    )
    features["PointsDifference"] = (
        features["HomePointsBefore"] - features["AwayPointsBefore"]
    )
 
    #Final feature set 
    features = features.merge(
        data[["Date", "Home Team", "Away Team", "Outcome"]],
        on=["Date", "Home Team", "Away Team"],
        how="left",
    )
 
    return features


def get_latest_team_stats(data):
    """
    Returns each team's cumulative stats through their most recent
    played match — used as the "before" stats for that team's NEXT
    (not yet played) fixture, for live predictions.
    """
    data = data.copy()
    goals = data["Result"].str.split("-", expand=True).astype(int)
    goals.columns = ["HomeGoals", "AwayGoals"]
    data["HomeGoals"] = goals["HomeGoals"]
    data["AwayGoals"] = goals["AwayGoals"]

    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data = data.sort_values("Date").reset_index(drop=True)

    home = data[["Date", "Home Team", "Away Team", "HomeGoals", "AwayGoals"]].copy()
    home["Team"] = home["Home Team"]
    home["GoalsScored"] = home["HomeGoals"]
    home["GoalsConceded"] = home["AwayGoals"]

    away = data[["Date", "Home Team", "Away Team", "HomeGoals", "AwayGoals"]].copy()
    away["Team"] = away["Away Team"]
    away["GoalsScored"] = away["AwayGoals"]
    away["GoalsConceded"] = away["HomeGoals"]

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values("Date").reset_index(drop=True)

    team_matches["Win"] = (team_matches["GoalsScored"] > team_matches["GoalsConceded"]).astype(int)
    team_matches["Points"] = 0
    team_matches.loc[team_matches["Win"] == 1, "Points"] = 3
    team_matches.loc[team_matches["GoalsScored"] == team_matches["GoalsConceded"], "Points"] = 1

    team_matches["MatchesPlayed"] = team_matches.groupby("Team").cumcount() + 1

    team_matches["AvgGoalsScored"] = (
        team_matches.groupby("Team")["GoalsScored"].transform(lambda x: x.cumsum())
        / team_matches["MatchesPlayed"]
    )
    team_matches["AvgGoalsConceded"] = (
        team_matches.groupby("Team")["GoalsConceded"].transform(lambda x: x.cumsum())
        / team_matches["MatchesPlayed"]
    )
    team_matches["Last5Goals"] = (
        team_matches.groupby("Team")["GoalsScored"]
        .transform(lambda x: x.rolling(5, min_periods=1).sum())
    )
    team_matches["Last5Conceded"] = (
        team_matches.groupby("Team")["GoalsConceded"]
        .transform(lambda x: x.rolling(5, min_periods=1).sum())
    )
    team_matches["WinRate"] = (
        team_matches.groupby("Team")["Win"].transform(lambda x: x.cumsum())
        / team_matches["MatchesPlayed"]
    )
    team_matches["PointsTotal"] = team_matches.groupby("Team")["Points"].transform(lambda x: x.cumsum())

    latest = team_matches.sort_values("Date").groupby("Team").tail(1).set_index("Team")

    return latest[[
        "AvgGoalsScored", "AvgGoalsConceded",
        "Last5Goals", "Last5Conceded",
        "WinRate", "PointsTotal", "MatchesPlayed",
    ]]


def get_latest_team_stats_safe(data):
    """Same as get_latest_team_stats, but returns an empty frame
    instead of erroring when there's no current-season data yet."""
    if data.empty:
        return pd.DataFrame(columns=[
            "AvgGoalsScored", "AvgGoalsConceded",
            "Last5Goals", "Last5Conceded",
            "WinRate", "PointsTotal", "MatchesPlayed",
        ])
    return get_latest_team_stats(data)

def get_live_team_stats(current_data, previous_season_data, current_teams, min_matches=5):
    current_stats = get_latest_team_stats_safe(current_data)
    prev_stats = get_latest_team_stats(previous_season_data)
    prev_stats["PointsPerGame"] = prev_stats["PointsTotal"] / prev_stats["MatchesPlayed"]

    league_avg = prev_stats[[
        "AvgGoalsScored", "AvgGoalsConceded",
        "Last5Goals", "Last5Conceded",
        "WinRate", "PointsPerGame",
    ]].mean()

    zero_row = pd.Series({
        "AvgGoalsScored": 0.0, "AvgGoalsConceded": 0.0,
        "Last5Goals": 0.0, "Last5Conceded": 0.0,
        "WinRate": 0.0, "PointsTotal": 0.0, "MatchesPlayed": 0,
    })

    rows = []
    for team in current_teams:
        row = current_stats.loc[team] if team in current_stats.index else zero_row
        matches_played = row["MatchesPlayed"]
        weight = min(matches_played / min_matches, 1.0)

        if team in prev_stats.index:
            baseline = prev_stats.loc[team]
            baseline_source = "last_season"
        else:
            baseline = league_avg
            baseline_source = "league_average"

        current_ppg = row["PointsTotal"] / matches_played if matches_played > 0 else 0

        blended = {
            "AvgGoalsScored": weight * row["AvgGoalsScored"] + (1 - weight) * baseline["AvgGoalsScored"],
            "AvgGoalsConceded": weight * row["AvgGoalsConceded"] + (1 - weight) * baseline["AvgGoalsConceded"],
            "Last5Goals": weight * row["Last5Goals"] + (1 - weight) * baseline["Last5Goals"],
            "Last5Conceded": weight * row["Last5Conceded"] + (1 - weight) * baseline["Last5Conceded"],
            "WinRate": weight * row["WinRate"] + (1 - weight) * baseline["WinRate"],
        }
        blended_ppg = weight * current_ppg + (1 - weight) * baseline["PointsPerGame"]
        blended["PointsTotal"] = blended_ppg * min_matches
        blended["MatchesPlayed"] = matches_played
        blended["IsEstimated"] = matches_played < min_matches
        blended["BaselineSource"] = baseline_source if matches_played < min_matches else "actual"
        rows.append(pd.Series(blended, name=team))

    return pd.DataFrame(rows)
