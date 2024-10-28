import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

def create_discrete_classes(data):
    """
    Convert Sleep_Quality into discrete classes.

    Args:
        data (pd.DataFrame): The processed sleep data.

    Returns:
        pd.DataFrame: The data with updated Sleep_Quality classes.
    """
    bins = [-1, 0.2, 0.6, 1.0]  # Define the bin edges
    labels = ['Low', 'Medium', 'High']  # Define the corresponding class labels

    # Create a new column with discrete classes based on Sleep_Quality
    data['Sleep_Quality'] = pd.cut(data['Sleep_Quality'], bins=bins, labels=labels)

    return data

def train_naive_bayes(data):
    """
    Train a Naive Bayes Classifier.

    Args:
        data (pd.DataFrame): The processed sleep data.

    Returns:
        GaussianNB: The trained model.
        pd.Series: The predictions made on the test set.
    """
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define the model
    model = GaussianNB()

    # Fit the model with resampled data
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

    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    print("\nCross-Validation Scores:")
    print(scores)
    print(f"Mean Cross-Validation Score: {scores.mean():.4f}")

if __name__ == "__main__":
    # Path to normalized data
    input_file_normalized = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

    # Load the normalized data
    normalized_data = load_data(input_file_normalized)

    # Convert Sleep_Quality to discrete classes
    normalized_data = create_discrete_classes(normalized_data)

    # Train the Naive Bayes model
    model, predictions, y_test = train_naive_bayes(normalized_data)

    # Evaluate the model's performance
    evaluate_model(y_test, predictions)

    # Cross-validate the model (optional)
    cross_validate_model(normalized_data, model)
