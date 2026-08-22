from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
 
def split_data(x, y, test_size=0.2, random_state=42):
    """
    Split features/labels into train and test sets.
    Stratified on y so H/D/A class balance is preserved in both splits.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test
 
 
def train_model(x_train, y_train, random_state=42):
    """
    Train a RandomForestClassifier on the given training data.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model
 
 
def make_predictions(model, x_test):
    """
    Return predicted class labels for the given feature set.
    """
    return model.predict(x_test)
 
 
def evaluate_model(y_test, predictions):
    """
    Print accuracy, a classification report, and a confusion matrix.
    Returns accuracy as a float.
    """
    accuracy = accuracy_score(y_test, predictions)
 
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows = actual, cols = predicted):")
    labels = sorted(set(y_test) | set(predictions))
    cm = confusion_matrix(y_test, predictions, labels=labels)
    print("Labels:", labels)
    print(cm)
 
    return accuracy
 
