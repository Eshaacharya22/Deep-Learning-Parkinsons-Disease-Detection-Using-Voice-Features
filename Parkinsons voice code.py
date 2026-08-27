# Parkinson's Voice Dataset Analysis
# VS Code Ready Python Script

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE


# ---------------- LOAD DATA ----------------
def load_data():
    # CSV must be in same folder
    df = pd.read_csv("parkinsons.csv")
    return df


# ---------------- PREPROCESS ----------------
def preprocess(df):
    X = df.drop("status", axis=1)
    y = df["status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    return train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )


# ---------------- TRAIN MODELS ----------------
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }

    print("\nModel Accuracy Results")
    print("-" * 30)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name}: {acc:.4f}")


# ---------------- MAIN ----------------
def main():
    print("Loading dataset...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess(df)

    print("Training models...")
    train_models(X_train, X_test, y_train, y_test)

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
