import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Load the sleep data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def train_decision_tree(data):
    """
    Train a Decision Tree Classifier on the binary 'Sleep_Quality'.

    Args:
        data (pd.DataFrame): The processed sleep data.

    Returns:
        DecisionTreeClassifier: The trained model.
        pd.Series: The predictions made on the test set.
    """
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = DecisionTreeClassifier(random_state=42)


    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    return model, predictions, y_test

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the trained model.

    Args:
        y_true (pd.Series): The true target values.
        y_pred (pd.Series): The predicted target values.
    """
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def cross_validate_model(data, model):
    """
    Perform cross-validation on the model using the provided data.

    Args:
        data (pd.DataFrame): The processed sleep data.
        model: The model to evaluate.
    """
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    scores = cross_val_score(model, X, y, cv=5)
    print("\nCross-Validation Scores:")
    print(scores)
    print(f"Mean Cross-Validation Score: {scores.mean():.4f}")

if __name__ == "__main__":

    input_file_normalized = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"


    normalized_data = load_data(input_file_normalized)


    model, predictions, y_test = train_decision_tree(normalized_data)


    evaluate_model(y_test, predictions)


    cross_validate_model(normalized_data, model)
