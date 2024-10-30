import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox


def load_data(file_path):
    """Load the sleep data from a CSV file."""
    return pd.read_csv(file_path)


def train_random_forest(data):
    """Train a Random Forest classifier."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    display_results(y, y_pred, "Random Forest")


def train_decision_tree(data):
    """Train a Decision Tree classifier."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    display_results(y, y_pred, "Decision Tree")


def train_naive_bayes(data):
    """Train a Naive Bayes classifier."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    nb = GaussianNB()
    nb.fit(X, y)
    y_pred = nb.predict(X)
    display_results(y, y_pred, "Naive Bayes")


def train_logistic_regression(data):
    """Train a Logistic Regression model."""
    X = data.drop(columns=['Sleep_Quality'])
    y = data['Sleep_Quality']

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X, y)
    y_pred = log_reg.predict(X)
    display_results(y, y_pred, "Logistic Regression")


def display_results(y_true, y_pred, model_name):
    """Display classification results including confusion matrix and report."""
    print(f"\nResults for {model_name}:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model_name}')

    # Ensure the plot is displayed
    plt.show()


def on_train_random_forest():
    """Handle training for Random Forest."""
    try:
        data = load_data(input_file)
        train_random_forest(data)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def on_train_decision_tree():
    """Handle training for Decision Tree."""
    try:
        data = load_data(input_file)
        train_decision_tree(data)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def on_train_naive_bayes():
    """Handle training for Naive Bayes."""
    try:
        data = load_data(input_file)
        train_naive_bayes(data)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def on_train_logistic_regression():
    """Handle training for Logistic Regression."""
    try:
        data = load_data(input_file)
        train_logistic_regression(data)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI setup
root = tk.Tk()
root.title("Sleep Quality Prediction Model Selector")

input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

model_label = tk.Label(root, text="Select a model to train:")
model_label.pack(pady=10)

# Create buttons for each model
rf_button = tk.Button(root, text="Train Random Forest", command=on_train_random_forest)
rf_button.pack(pady=5)

dt_button = tk.Button(root, text="Train Decision Tree", command=on_train_decision_tree)
dt_button.pack(pady=5)

nb_button = tk.Button(root, text="Train Naive Bayes", command=on_train_naive_bayes)
nb_button.pack(pady=5)

log_reg_button = tk.Button(root, text="Train Logistic Regression", command=on_train_logistic_regression)
log_reg_button.pack(pady=5)

root.mainloop()
