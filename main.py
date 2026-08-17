import pandas as pd 
from model import split_data

#load the dataset
data = pd.read_csv("2025_2026 dataset.csv")

'''Data Verifcation
print(data.head())
print(data.shape)
print(data.columns)

Write date in the correct format
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

print(data.dtypes)

Verify no data missing
print(data.isnull().sum())'''

#split one column into 2
goals = data["Result"].str.split(" - ", expand=True)
goals = goals.astype(int)

#Rename the columns
goals.columns = ["HomeGoals", "AwayGoals"]

#
data["HomeGoals"] = goals["HomeGoals"]
data["AwayGoals"] = goals["AwayGoals"]

#create a totalgoals column
data["TotalGoals"] = data["HomeGoals"] + data["AwayGoals"]
#print(data[["Home Team", "Away Team", "HomeGoals", "AwayGoals", "TotalGoals"]].head())

data["Outcome"] = "D"

data.loc[data["HomeGoals"] > data["AwayGoals"], "Outcome"] = "H"
data.loc[data["HomeGoals"] < data["AwayGoals"], "Outcome"] = "A"
#print(data[["Home Team", "Away Team", "HomeGoals", "AwayGoals", "Outcome"]].head(10))

#Team stats
home_goals = data.groupby("Home Team")["HomeGoals"].sum()
#print(home_goals.sort_values(ascending=False))

away_goals = data.groupby("Away Team")["AwayGoals"].sum()
#print(away_goals.sort_values(ascending=False))

team_goals = pd.DataFrame({
    "HomeGoals": home_goals,
    "AwayGoals": away_goals
})
#print(team_goals)

team_goals["TotalGoals"] = (
    team_goals["HomeGoals"] + team_goals["AwayGoals"]
)
#print(team_goals.sort_values("TotalGoals", ascending=False))

#Goals conceded
home_conceded = data.groupby("Home Team")["AwayGoals"].sum()
#print(home_conceded.sort_values(ascending=False))

away_conceded = data.groupby("Away Team")["HomeGoals"].sum()
#print(away_conceded.sort_values(ascending=False))

team_conceded = pd.DataFrame({
    "HomeConceded": home_conceded,
    "AwayConceded": away_conceded
})
#print(team_conceded)

team_conceded["TotalConceded"] = (
    team_conceded["HomeConceded"] + team_conceded["AwayConceded"]
)
#print(team_conceded.sort_values("TotalConceded", ascending=False))

home_wins = data[data["Outcome"] == "H"].groupby("Home Team").size()
#print(home_wins.sort_values(ascending=False))

home_draws = data[data["Outcome"] == "D"].groupby("Home Team").size()
#print(home_draws.sort_values(ascending=False))

away_wins = data[data["Outcome"] == "A"].groupby("Away Team").size()
#print(away_wins.sort_values(ascending=False))

away_draws = data[data["Outcome"] == "D"].groupby("Away Team").size()
##=print(away_draws.sort_values(ascending=False))

home_results = data.groupby("Home Team")["Outcome"].value_counts()
#print(home_results)

away_results = data.groupby("Away Team")["Outcome"].value_counts()
#print(away_results)

home_record = home_results.unstack(fill_value=0)
#print(home_record)

home_record = home_record.rename(columns = {
    "H": "HomeWins",
    "D": "HomeDraws",
    "A": "HomeLosses"
})

away_record = home_results.unstack(fill_value=0)
#print(away_record)

away_record = away_record.rename(columns = {
    "H": "AwayWins",
    "D": "AwayDraws",
    "A": "AwayLosses"
})

team_record = pd.concat([home_record, away_record], axis=1)
team_record.columns.name = None
team_record.index.name = None

team_record = team_record[
    [
        "HomeWins",
        "HomeDraws",
        "HomeLosses",
        "AwayWins",
        "AwayDraws",
        "AwayLosses"
    ]
]

#print(team_record)

#Phase 2

liverpool = data[
    (data["Home Team"] == "Liverpool") | 
    (data["Away Team"] == "Liverpool")
    ]

liverpool["GoalsScored"] = liverpool.apply(
    lambda row: row["HomeGoals"]
    if row["Home Team"] == "Liverpool"
    else row["AwayGoals"],
    axis=1
)
liverpool["GoalsBefore"] = liverpool["GoalsScored"].cumsum().shift(1).fillna(0)

liverpool["GoalsConceded"] = liverpool.apply(
    lambda row: row["AwayGoals"]
    if row["Home Team"] == "Liverpool"
    else row["HomeGoals"],
    axis=1
)
liverpool["ConcededBefore"] = liverpool["GoalsConceded"].cumsum().shift(1).fillna(0)

data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data = data.sort_values("Date")
#print(data[["Date", "Home Team", "Away Team"]].head())
#print(data["Date"].dtype)

home = data[[
    "Date",
    "Home Team",
    "Away Team",
    "HomeGoals",
    "AwayGoals"
]].copy()

home["Team"] = home["Home Team"]
home["Opponent"] = home["Away Team"]
home["GoalsScored"] = home["HomeGoals"]
home["GoalsConceded"] = home["AwayGoals"]

away = data[[
    "Date",
    "Home Team",
    "Away Team",
    "HomeGoals",
    "AwayGoals"
]].copy()

away["Team"] = away["Away Team"]
away["Opponent"] = away["Home Team"]
away["GoalsScored"] = away["AwayGoals"]
away["GoalsConceded"] = away["HomeGoals"]

team_matches =pd.concat([home, away], ignore_index=True)

team_matches.groupby("Team")

team_matches["GoalsBefore"] = (
    team_matches.groupby("Team")["GoalsScored"].transform(lambda x: x.cumsum().shift(1)).fillna(0)
)

team_matches["ConcededBefore"] = (
    team_matches.groupby("Team")["GoalsConceded"].transform(lambda x: x.cumsum().shift(1)).fillna(0)
)

team_matches["MatchesBefore"] = (
    team_matches.groupby("Team").cumcount()
)

team_matches["AvgGoalsBefore"] = (
    team_matches["GoalsBefore"] / team_matches["MatchesBefore"]
).fillna(0)

team_matches["AvgConcededBefore"] = (
    team_matches["ConcededBefore"] / team_matches["MatchesBefore"]
).fillna(0)

team_matches["Last5Goals"] = (
    team_matches.groupby("Team")["GoalsScored"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    .fillna(0)
)

team_matches["Last5Conceded"] = (
    team_matches.groupby("Team")["GoalsConceded"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    .fillna(0)
)

'''print(
    team_matches[team_matches["Team"] == "Liverpool"][
        [
            "Date",
            "Team",
            "Opponent",
            "GoalsScored",
            "GoalsConceded",
            "GoalsBefore",
            "ConcededBefore",
            "MatchesBefore",
            "AvgGoalsBefore",
            "AvgConcededBefore",
            "Last5Goals",
            "Last5Conceded"
        ]
    
].head(10))'''

home_features = team_matches[
    team_matches["Team"] == team_matches["Home Team"]
].copy()

away_features = team_matches[
    team_matches["Team"] == team_matches["Away Team"]
].copy()

home_features = home_features.rename(
    columns={
        "AvgGoalsBefore": "HomeAvgGoalsBefore",
        "AvgConcededBefore": "HomeAvgConcededBefore",
        "Last5Goals": "HomeLast5Goals",
        "Last5Conceded": "HomeLast5Conceded"
    }
)

away_features = away_features.rename(
    columns={
        "AvgGoalsBefore": "AwayAvgGoalsBefore",
        "AvgConcededBefore": "AwayAvgConcededBefore",
        "Last5Goals": "AwayLast5Goals",
        "Last5Conceded": "AwayLast5Conceded"
    }
)


data = data.merge(
    home_features[
        [
            "Date",
            "Home Team",
            "Away Team",
            "HomeAvgGoalsBefore",
            "HomeAvgConcededBefore",
            "HomeLast5Goals",
            "HomeLast5Conceded"
        ]
    ],
    on = ["Date", "Home Team", "Away Team"],
    how = "left"
)

data = data.merge(
    away_features[
        [
            "Date",
            "Home Team",
            "Away Team",
            "AwayAvgGoalsBefore",
            "AwayAvgConcededBefore",
            "AwayLast5Goals",
            "AwayLast5Conceded"
        ]
    ],
    on = ["Date", "Home Team", "Away Team"],
    how = "left"
)

data["AttackDifference"] = (
    data["HomeAvgGoalsBefore"] - data["AwayAvgGoalsBefore"]
)

data["DefenseDifference"] = (
    data["AwayAvgConcededBefore"] - data["HomeAvgConcededBefore"]
)

data["FormDifference"] = (
    data["HomeLast5Goals"] - data["AwayLast5Goals"]
)

data["FormDefenseDifference"] = (
    data["AwayLast5Conceded"] - data["HomeLast5Conceded"]
)

'''print(
    data[
        [
            "Home Team",
            "Away Team",
            "AttackDifference",
            "DefenseDifference",
            "FormDifference",
            "FormDefenseDifference"
        ]
    ].head(10)
)'''

x = data[
    [
        "AttackDifference",
        "DefenseDifference",
        "FormDifference",
        "FormDefenseDifference"
    ]
]

y = data["Outcome"]

x_train, x_test, y_train, y_test = split_data(x, y)

