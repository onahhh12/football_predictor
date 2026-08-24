import pandas as pd
from features import create_features
from model import split_data, train_model, make_predictions, evaluate_model
 
# Load the dataset
data = pd.read_csv("2026_2027 dataset.csv")
 
#Create features (includes Outcome label)
features = create_features(data)
 
print(features.shape)
print(features.columns.tolist())
 
#Machine learning phase
x = features[
    [
        "AttackDifference",
        "DefenseDifference",
        "FormDifference",
        "FormDefenseDifference",
        "PointsDifference"
    ]
]
y = features["Outcome"]
 
x_train, x_test, y_train, y_test = split_data(x, y)
model = train_model(x_train, y_train)
predictions = make_predictions(model, x_test)
accuracy = evaluate_model(y_test, predictions)
 
