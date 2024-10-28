import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: No data found in the file.")
        raise


def create_discrete_classes(data):
    """
    Convert Sleep_Quality into discrete classes.

    Args:
        data (pd.DataFrame): The processed sleep data.

    Returns:
        pd.DataFrame: The data with updated Sleep_Quality classes.
    """
    # Define bins and labels for the Sleep_Quality classes
    bins = [-1, 0.2, 0.6, 1.0]  # Define the bin edges
    labels = ['Low', 'Medium', 'High']  # Define the corresponding class labels

    # Create a new column with discrete classes based on Sleep_Quality
    data['Sleep_Quality'] = pd.cut(data['Sleep_Quality'], bins=bins, labels=labels)

    return data


def train_knn_with_tuning_and_smote(data):
    """
    Train a K-Nearest Neighbors Classifier with hyperparameter tuning and SMOTE for class imbalance.

    Args:
        data (pd.DataFrame): The processed sleep data.

    Returns:
        KNeighborsClassifier: The trained model.
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
    model = KNeighborsClassifier()

    # Set the parameters for GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='f1_macro', cv=5, n_jobs=-1)

    # Fit the model with resampled data
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Best model from GridSearch
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    predictions = best_model.predict(X_test)

    return best_model, predictions, y_test


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

    # Train the KNN model with hyperparameter tuning and SMOTE
    model, predictions, y_test = train_knn_with_tuning_and_smote(normalized_data)

    # Evaluate the model's performance
    evaluate_model(y_test, predictions)

    # Cross-validate the model (optional)
    cross_validate_model(normalized_data, model)
