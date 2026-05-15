import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
path = "../Data/predictive_maintenance.csv"
data = pd.read_csv(path, delimiter=',')

# Drop identifiers and columns that cause data leakage
data.drop(['Failure Type', 'UDI'], axis=1, inplace=True)

# Separate target and features
y = data['Target']
X = data.drop(['Target'], axis=1)

# Encode categorical features
categorical_cols = ['Product ID', 'Type']
X[categorical_cols] = X[categorical_cols].astype('category')
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.cat.codes)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False
)

# Train model
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'XGBoost Test Accuracy: {accuracy:.4f}')
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))