import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the sleep data from a CSV file."""
    return pd.read_csv(file_path)

def train_logistic_regression(data):
    """Train a Logistic Regression model with cross-validation and analyze results."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    # Initialize the logistic regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated accuracy
    cv_scores = cross_val_score(log_reg, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Fit the model on the full dataset
    log_reg.fit(X, y)
    y_pred = log_reg.predict(X)

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

    return log_reg

if __name__ == "__main__":
    """Load data, train the Logistic Regression model, and print the results."""
    input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

    data = load_data(input_file)
    best_model = train_logistic_regression(data)
