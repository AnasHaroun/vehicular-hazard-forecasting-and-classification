# -*- coding: utf-8 -*-
"""
Stacked hazard prediction model script
Originally based on Colab notebook
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from google.colab import drive

# Mount Drive and load CSV files
drive.mount('/content/drive')
data_dir = "/content/drive/MyDrive/Prediction Datasets/Csv6.3"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# Fill missing values
df.fillna({
    'hazardAttack': 0,
    'hazardOccurrence': -1,
    'hazardOccurrencePercentage': 0
}, inplace=True)

# Select subset of columns
dfAll = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,12,13,14,19,21,22,20]]
df.index = df['rcvTime']
dfAll.index = dfAll['rcvTime']

# Select a time window
hazaAndTime = dfAll[(df['rcvTime'] >= 200) & (df['rcvTime'] <= 4200)]

# Plot hazard occurrence and attack
mpl.rcParams['agg.path.chunksize'] = 10000
hazaAndTime['hazardOccurrence'].plot(figsize=(13, 9))
plt.xlabel("Time (in sec.)")
plt.ylabel("Hazard Occurrence")
plt.title("Hazard Occurrence Over Time")
plt.yticks([-1, 0, 1])
plt.show()

hazaAndTime['hazardAttack'].plot(figsize=(13, 9))
plt.xlabel("Time (in sec.)")
plt.ylabel("Hazard Attack")
plt.title("Hazard Attack Over Time")
plt.yticks([-1, 0, 1])
plt.show()


# --- Sequence Preparation ---
def df_to_X_y(df, window_size=5, step=10):
    df_np = df.to_numpy()
    X, y = [], []
    for i in range(0, len(df_np) - window_size, step):
        row = [[a] for a in df_np[i:i + window_size]]
        X.append(row)
        y.append(df_np[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
STEP = 1
X, y = df_to_X_y(hazaAndTime['hazardOccurrence'], WINDOW_SIZE, STEP)

# --- Train/Test/Validation Split ---
dataset_size = len(y)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# --- LSTM Model ---
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Nadam

model1 = Sequential([
    InputLayer((WINDOW_SIZE, 1)),
    LSTM(WINDOW_SIZE, name='LSTM_1'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model1.compile(
    loss=MeanSquaredError(),
    optimizer=Nadam(learning_rate=0.001),
    metrics=[RootMeanSquaredError()]
)

checkpoint = ModelCheckpoint('model1/', save_best_only=True)
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, callbacks=[checkpoint])

# Load best model and generate predictions
model1 = load_model('model1/')
train_predictions = model1.predict(X_train).flatten()
val_predictions = model1.predict(X_val).flatten()
test_predictions = model1.predict(X_test).flatten()


import sklearn.metrics as sm

# --- Evaluate LSTM Model ---
print("Train R2 score:", round(sm.r2_score(y_train, train_predictions), 2))
print("Validation R2 score:", round(sm.r2_score(y_val, val_predictions), 2))
print("Test R2 score:", round(sm.r2_score(y_test, test_predictions), 2))

# --- Plot Test Predictions ---
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(test_predictions[:100], label='Predicted')
plt.plot(y_test[:100], label='Actual')
plt.title("Test Prediction vs Actual")
plt.legend()
plt.show()

# --- Prepare data for classification ---
hazaAndTime.drop(['EventID', 'sendTime', 'RoadID'], axis=1, inplace=True)
hazaAndTime.fillna({
    'lanePosition': 0,
    'sender': 0,
    'messageID': 0,
    'maxDeceleration': 0
}, inplace=True)

# Convert to numpy arrays
Xc = hazaAndTime.to_numpy()[:, :-1]
yc = hazaAndTime.to_numpy()[:, -1]

Xc_train = Xc[:train_size]
yc_train = yc[:train_size]
Xc_val = Xc[train_size:train_size + val_size]
yc_val = yc[train_size:train_size + val_size]
Xc_test = Xc[train_size + val_size:]
yc_test = yc[train_size + val_size:]

# --- Logistic Regression Classification ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

clf = LogisticRegression(class_weight="balanced")
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

# Evaluation
print("Classification Accuracy:", accuracy_score(yc_test, yc_pred))
print("F1 Score:", f1_score(yc_test, yc_pred))
print("Confusion Matrix:\n", confusion_matrix(yc_test, yc_pred))

# Class distribution
unique_elements, counts_elements = np.unique(yc_test, return_counts=True)
print("Label distribution in test set:", dict(zip(unique_elements, counts_elements)))

# --- Integrating LSTM Predictions into Stacked Model ---

# Prepare combined classification + prediction arrays
Xcp = hazaAndTime.to_numpy()[:, :-1]
ycp = hazaAndTime.to_numpy()[:, -1]

Xcp_train = Xcp[:train_size]
ycp_train = ycp[:train_size]
Xcp_val = Xcp[train_size:train_size + val_size]
ycp_val = ycp[train_size:train_size + val_size]
Xcp_test = Xcp[train_size + val_size:]
ycp_test = ycp[train_size + val_size:]

# Insert train predictions into a new feature column
Xcp_train = np.insert(Xcp_train, 11, train_predictions, axis=1)

# Handle test prediction length mismatch due to sequence offset
WINDOW_SIZE = 30  # Reaffirm for consistency
test_predictions_padded = np.append(np.full(WINDOW_SIZE, 0), test_predictions)
Xcp_test = np.insert(Xcp_test, 11, test_predictions_padded, axis=1)

# Same insert for classification data
Xc_train = np.insert(Xc_train, 13, train_predictions, axis=1)
Xc_test = np.insert(Xc_test, 13, test_predictions_padded, axis=1)

# --- Classification with Stacked Model ---
clf_stacked = LogisticRegression(class_weight="balanced")
clf_stacked.fit(Xc_train, yc_train)
yc_pred_stacked = clf_stacked.predict(Xc_test)

# Evaluation
print("Stacked Model Accuracy:", accuracy_score(yc_test, yc_pred_stacked))
print("Stacked Model F1 Score:", f1_score(yc_test, yc_pred_stacked))
print("Stacked Confusion Matrix:\n", confusion_matrix(yc_test, yc_pred_stacked))

# Class distribution
unique_elements, counts_elements = np.unique(yc_test, return_counts=True)
print("Label distribution in test set (stacked):", dict(zip(unique_elements, counts_elements)))


# --- Optional: Plot Stacked Model Predictions ---
plt.figure(figsize=(10, 5))
plt.plot(yc_pred_stacked[:100], label='Stacked Predictions')
plt.plot(yc_test[:100], label='Actual Labels')
plt.title("Stacked Model Predictions vs Actual Labels")
plt.xlabel("Sample Index")
plt.ylabel("Hazard Attack (Binary)")
plt.legend()
plt.show()

# --- Optional: Exporting Results to CSV ---
results_df = pd.DataFrame({
    'Actual': yc_test,
    'Stacked_Predicted': yc_pred_stacked
})
results_df.to_csv("stacked_model_predictions.csv", index=False)
print("Predictions exported to 'stacked_model_predictions.csv'")

# --- Optional: Save Trained Models (if needed) ---
import joblib

# Save the stacked classifier
joblib.dump(clf_stacked, "logistic_stacked_model.pkl")
print("Classifier saved to 'logistic_stacked_model.pkl'")

# Save LSTM model if not already saved
# model1.save('model1/') was done earlier with checkpoint
