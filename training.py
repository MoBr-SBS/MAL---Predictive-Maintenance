import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

path = "predictive_maintenance.csv"
data = pd.read_csv(path, delimiter=',')
print(data.head())
print("Empty columns: ", data.columns[data.isnull().any()])

# Delete "Failure Type" to prevent Data-Leakage
# Delete "UDI" as it is only a Data-ID
data.drop(['Failure Type', 'UDI'], axis = 1, inplace=True)

# Select "Target" as value to be predicted
col = data['Target']
col = pd.get_dummies(col, dtype=float)
data = data.drop(['Target'], axis = 1)

# converting strings to numbers
conv_num = ['Product ID','Type']
data[conv_num] = data[conv_num].astype('category')
data[conv_num] = data[conv_num].apply(lambda x: x.cat.codes)

#print(data.shape)
#print(data.shape[1])

# Erzeuge Objekte
s_scaler = StandardScaler()

# Spalten für StandardScaler
cols_to_s_scale = ['Air temperature K','Process temperature K','Rotational speed rpm','Torque Nm','Tool wear min']
data[cols_to_s_scale] = s_scaler.fit_transform(data[cols_to_s_scale])

#print(data.head())

# Aus den zwei Tabellen vier Tabellen erzeugen
train_data, test_data, train_col, test_col = train_test_split(data,col, test_size=0.2, random_state=42)

# Aufbau KNN
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(data.shape[1],)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

# Konfiguration des Lernprozesses
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 10 Durchläufe
model.fit(train_data, train_col, epochs=10)

test_loss, test_acc = model.evaluate(test_data, test_col)
print('Test accuracy:', test_acc)