import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load the sleep data from a CSV file."""
    return pd.read_csv(file_path)


def train_naive_bayes_with_cv(data):
    """Train a Naive Bayes classifier with cross-validation and analyze results."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    # Naive Bayes does not require hyperparameter tuning like other classifiers,
    # but you can create a simple cross-validation setup.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gnb = GaussianNB()

    # Cross-validated accuracy
    cv_scores = cross_val_score(gnb, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Fit the model on the full dataset
    gnb.fit(X, y)
    y_pred = gnb.predict(X)
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

    return gnb


if __name__ == "__main__":
    """Load data, train the Naive Bayes classifier, and print the results."""
    input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

    data = load_data(input_file)
    best_model = train_naive_bayes_with_cv(data)
