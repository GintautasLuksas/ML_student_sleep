import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def load_data(file_path):
    """Load the sleep data from a CSV file."""
    return pd.read_csv(file_path)


def check_imbalance(data):
    """Check for class imbalance in the dataset."""
    class_counts = data['Sleep_Quality'].value_counts()
    print("Class distribution:\n", class_counts)


def train_random_forest_with_cv(data):
    """Train a Random Forest classifier with hyperparameter tuning and cross-validation."""
    # Separate features and target variable
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    # Check for data imbalance
    check_imbalance(data)

    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=cv, scoring='accuracy',
                               n_jobs=-1, verbose=1)

    # Fit the model
    grid_search.fit(X, y)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best parameters: {best_params}")

    # Evaluate the best model with cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=cv)
    print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    return best_model


if __name__ == "__main__":
    """Load data, train the Random Forest classifier with hyperparameter tuning, and print the results."""
    input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

    # Load the normalized data
    data = load_data(input_file)

    # Train the classifier with hyperparameter tuning and cross-validation
    best_model = train_random_forest_with_cv(data)
