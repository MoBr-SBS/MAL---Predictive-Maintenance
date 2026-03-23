import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

path = "predictive_maintenance.csv"
data = pd.read_csv(path, delimiter=',')

# Delete "Failure Type" to prevent Data-Leakage
# Delete "UDI" as it is only a Data-ID
data.drop(['Failure Type', 'UDI'], axis = 1, inplace=True)

# WICHTIG: Für XGBoost behalten wir das Target als eine Spalte (0 oder 1)
# Statt get_dummies nehmen wir einfach die Werte direkt
y = data['Target']
X = data.drop(['Target'], axis = 1)

# Kategorische Strings in Zahlen umwandeln
conv_num = ['Product ID', 'Type']
X[conv_num] = X[conv_num].astype('category')
X[conv_num] = X[conv_num].apply(lambda x: x.cat.codes)

# Datensatz aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- XGBOOST MODEL ---

# Initialisierung des Classifiers
# Du kannst hier Hyperparameter wie max_depth oder n_estimators direkt setzen
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False # Verhindert Warnungen in neueren Versionen
)

# Training
model.fit(X_train, y_train)

# Vorhersage & Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'XGBoost Test Accuracy: {accuracy:.4f}')
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))