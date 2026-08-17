from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
    )
    model.fit(x_train, y_train)
    return model

def make_predictions(model, x_test):
    predictions = model.predict(x_test)
    return predictions

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)

    return accuracy