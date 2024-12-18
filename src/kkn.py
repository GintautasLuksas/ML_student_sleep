import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load the sleep data from a CSV file."""
    return pd.read_csv(file_path)


def train_knn_with_cv(data):
    """Train a KNN classifier with hyperparameter tuning and cross-validation, and analyze results."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier()

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,
                               cv=cv, scoring='accuracy',
                               n_jobs=-1, verbose=1)

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    cv_scores = cross_val_score(best_model, X, y, cv=cv)
    print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    print("Final Classification Report on Full Dataset:\n")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    return best_model


if __name__ == "__main__":
    """Load data, train the KNN classifier with hyperparameter tuning, and print the results."""
    input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

    data = load_data(input_file)
    best_model = train_knn_with_cv(data)
