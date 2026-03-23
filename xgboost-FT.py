import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Daten laden
path = "predictive_maintenance.csv"
data = pd.read_csv(path, delimiter=',')

# 2. Datenbereinigung
# 'Target' und 'UDI' entfernen (verhindert Data-Leakage und Rauschen)
data.drop(['Target', 'UDI', 'Product ID'], axis=1, inplace=True)

# 3. Zielvariable (y) vorbereiten
# XGBoost benötigt numerische Labels (0, 1, 2...). Der LabelEncoder erledigt das.
le = LabelEncoder()
y = le.fit_transform(data['Failure Type'])

# Zeigt dir an, welche Zahl für welchen Fehler steht (hilfreich zur Kontrolle)
mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(f"Mapping der Fehler-Typen: {mapping}\n")

# 4. Features (X) vorbereiten
X = data.drop(['Failure Type'], axis=1)

# Kategorische Strings in X (wie Product ID und Type) in Zahlen umwandeln
conv_num = ['Type']
X[conv_num] = X[conv_num].astype('category')
X[conv_num] = X[conv_num].apply(lambda x: x.cat.codes)

# 5. Datensatz aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. XGBoost Modell initialisieren
# 'multi:softprob' oder 'multi:softmax' wird von XGBoost automatisch gewählt,
# wenn y mehr als zwei Kategorien hat.
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 7. Training
model.fit(X_train, y_train)

# 8. Vorhersage & Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'XGBoost Test Accuracy: {accuracy:.4f}')
print("-" * 30)
print("Detaillierter Bericht:")
# target_names sorgt dafür, dass im Bericht wieder 'Heat Dissipation' etc. steht
print(classification_report(y_test, y_pred, target_names=le.classes_))