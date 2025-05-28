import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load and concatenate all CSV files
data_dir = "/content/drive/MyDrive/Prediction Datasets/PredictionDataset2"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# Drop every second row to downsample
df = df.iloc[::2]

# Fill missing values
df['hazardAttack'] = df['hazardAttack'].fillna(-1)
df['messageID'] = df['messageID'].fillna(0)

# Select relevant columns (excluding EventID)
df = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,12,13,14,19]]

# Set time as index
df.index = df['rcvTime']

# Visualize hazard attack activity
hazAttck = df['hazardAttack']
hazAttck.plot()
plt.xlim(2400, 3200)
plt.title("Hazard Attack Timeline")
plt.show()

# Focus on a time window
hazaAndTime = df[(df['rcvTime'] >= 2400) & (df['rcvTime'] <= 3200)]
hazaAndTime.plot(figsize=(20,10))
plt.title("Windowed Hazard Attack Timeline")
plt.show()

################################################ Part 2: Time-Series Preparation and Dataset Splitting

# Function to convert a time series to supervised learning format
def df_to_X_y(df, window_size=60, step=10):
    df_np = df.to_numpy()
    X, y = [], []
    for i in range(0, len(df_np) - window_size, step):
        X.append([[a] for a in df_np[i:i + window_size]])
        y.append(df_np[i + window_size])
    return np.array(X), np.array(y)

# Apply to hazard attack series
WINDOW_SIZE = 60
STEP = 10
X, y = df_to_X_y(hazAttck, WINDOW_SIZE, STEP)

# Split into train/val/test sets
dataset_size = len(y)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)

X_train, y_train = X[:train_size], y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print("Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


################################################ Part 3: GRU Model Building and Training 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, GRU, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Nadam

# Build the model
model = Sequential([
    InputLayer((WINDOW_SIZE, 1)),
    GRU(WINDOW_SIZE, return_sequences=True, name='GRU_1'),
    GRU(WINDOW_SIZE, name='GRU_2'),
    Dense(64, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.summary()

# Compile the model
model.compile(
    loss=MeanSquaredError(),
    optimizer=Nadam(learning_rate=0.001),
    metrics=[RootMeanSquaredError(), "accuracy"]
)

# Save the best model using validation loss
checkpoint = ModelCheckpoint('model1/', save_best_only=True)

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    verbose=1,
    callbacks=[checkpoint]
)


################################################  Part 4: Model Evaluation and Visualization


from tensorflow.keras.models import load_model
import sklearn.metrics as sm

# Load the best saved model
model = load_model('model1/')

# --- Evaluate on Training Data ---
train_predictions = model.predict(X_train[1:1001]).flatten()
train_actuals = y_train[1:1001]

print("Train R² Score:", round(sm.r2_score(train_actuals, train_predictions), 2))

# Plot training predictions vs actual
plt.figure(figsize=(10, 4))
plt.plot(train_predictions, label="Predicted")
plt.plot(train_actuals, label="Actual")
plt.title("Training Data: Predictions vs Actuals")
plt.legend()
plt.show()

# --- Evaluate on Validation Data ---
val_predictions = model.predict(X_val).flatten()
print("Validation R² Score:", round(sm.r2_score(y_val, val_predictions), 2))

plt.figure(figsize=(10, 4))
plt.plot(val_predictions[:100], label="Predicted")
plt.plot(y_val[:100], label="Actual")
plt.title("Validation Data: Predictions vs Actuals")
plt.legend()
plt.show()

# --- Evaluate on Test Data ---
test_predictions = model.predict(X_test).flatten()
print("Test R² Score:", round(sm.r2_score(y_test, test_predictions), 2))

plt.figure(figsize=(10, 4))
plt.plot(test_predictions[:100], label="Predicted")
plt.plot(y_test[:100], label="Actual")
plt.title("Test Data: Predictions vs Actuals")
plt.legend()
plt.show()

################################################  Part 5: Optional Cleanup and Export

# --- Optional: Save predictions for external analysis ---

import pandas as pd

# Save test results to CSV
test_results = pd.DataFrame({
    'Test Predictions': test_predictions,
    'Actuals': y_test
})
test_results.to_csv("gru_test_predictions.csv", index=False)
print("Test predictions saved to 'gru_test_predictions.csv'.")

# Save the trained model (redundant if already saved during checkpointing)
# model.save('final_gru_model.h5')  # Uncomment if needed

# --- Optional: Summary statistics ---
print("\nSummary Statistics:")
print("Train MAE:", round(sm.mean_absolute_error(train_actuals, train_predictions), 2))
print("Validation MAE:", round(sm.mean_absolute_error(y_val, val_predictions), 2))
print("Test MAE:", round(sm.mean_absolute_error(y_test, test_predictions), 2))
