# -----------------------------------------------------------------------------
# DISCLAIMER: This code was developed with the assistance of Google Gemini.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


def run_predictive_maintenance_pipeline(csv_path):
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # 1. Exploratory Data Analysis (EDA) - Class Distribution
    print("Class Distribution in Dataset:")
    print(data['Failure Type'].value_counts())
    print("-" * 40)

    # 2. Feature Engineering (Domain-specific domain knowledge)
    data['Temp_Diff'] = data['Process temperature K'] - data['Air temperature K']
    data['Power'] = data['Torque Nm'] * data['Rotational speed rpm']
    # Interaction term for cumulative degradation
    data['Wear_Power_Interaction'] = data['Tool wear min'] * data['Power']

    # 3. Data Cleaning & Label Encoding
    drop_cols = ['UDI', 'Product ID', 'Target']
    data_cleaned = data.drop(columns=[col for col in drop_cols if col in data.columns])

    le = LabelEncoder()
    y = le.fit_transform(data_cleaned['Failure Type'])
    class_names = le.classes_

    X = data_cleaned.drop('Failure Type', axis=1)
    X['Type'] = X['Type'].astype('category').cat.codes

    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Addressing Class Imbalance via SMOTE (Synthetic Minority Over-sampling Technique)
    min_samples = pd.Series(y_train).value_counts().min()
    k_neighbors = min(3, min_samples - 1) if min_samples > 1 else 1

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Model Definition & Training (XGBoost Classifier)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        objective='multi:softprob'
    )
    model.fit(X_train_res, y_train_res)

    # 7. Model Inference
    y_pred = model.predict(X_test)

    # 8. Evaluation Metrics
    print(f"\n--- Model Evaluation Report ---")
    print(f"Global Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 9. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    # 10. Global Feature Importance (Model Explainability)
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top Feature Importances')
    plt.xlabel('Gain/Score')
    plt.show()


if __name__ == "__main__":
    run_predictive_maintenance_pipeline("data/predictive_maintenance.csv")