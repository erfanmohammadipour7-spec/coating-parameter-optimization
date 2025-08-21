# train.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # <-- ADDED METRICS
import matplotlib.pyplot as plt # <-- ADDED FOR PLOTTING
import seaborn as sns # <-- ADDED FOR PLOTTING

# Import from your other files
from data_utils import augment_data_with_noise, CustomDataset
from model import MLP

# --- 1. Configuration ---
FILE_PATH = 'data.csv'
INPUT_COLS = ['Current Density (mA/cm²)', 'Voltage (V)']
OUTPUT_COLS = ['Porosity (%)', 'Pore Size (µm)']

# Augmentation Settings
NOISE_LEVEL = 0.05
AUGMENTATION_FACTOR = 15

# Model & Training Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 300
BATCH_SIZE = 8
HIDDEN_NEURONS = 16
DROPOUT_RATE = 0.2

# --- 2. Load Original Data ---
try:
     original_df = pd.read_csv(FILE_PATH, encoding='latin-1')
except FileNotFoundError:
    print(f"Error: The data file '{FILE_PATH}' was not found.")
    exit()

print(f"Loaded {len(original_df)} original samples.")

X = original_df[INPUT_COLS]
y = original_df[OUTPUT_COLS]

# --- 3. Leave-One-Out Cross-Validation (LOOCV) ---
loo = LeaveOneOut()
all_scores = []
fold_counter = 1

# --- NEW: Lists to store all predictions and actual values ---
all_y_test = []
all_y_pred = []
# ---

print("\nStarting Leave-One-Out Cross-Validation...")

for train_index, test_index in loo.split(X):
    print(f"\n--- Fold {fold_counter}/{len(original_df)} ---")
    
    # Split data
    X_train_orig, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_orig, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Augment ONLY the training data
    df_to_augment = pd.concat([X_train_orig, y_train_orig], axis=1)
    augmented_part = augment_data_with_noise(df_to_augment, INPUT_COLS, NOISE_LEVEL, AUGMENTATION_FACTOR)
    
    # Combine original training with augmented data
    X_train_aug = pd.concat([X_train_orig, augmented_part[INPUT_COLS]], ignore_index=True)
    y_train_aug = pd.concat([y_train_orig, augmented_part[OUTPUT_COLS]], ignore_index=True)
    print(f"Training on {len(X_train_orig)} original + {len(augmented_part)} augmented samples = {len(X_train_aug)} total.")

    # Feature Scaling
    scaler = StandardScaler().fit(X_train_orig)
    X_train_scaled = scaler.transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    # Create PyTorch Datasets and DataLoader
    train_dataset = CustomDataset(X_train_scaled, y_train_aug.values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Model Training ---
    model = MLP(input_size=len(INPUT_COLS), output_size=len(OUTPUT_COLS), hidden_neurons=HIDDEN_NEURONS, dropout_rate=DROPOUT_RATE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()

    score = mean_absolute_error(y_test, y_pred)
    all_scores.append(score)
    
    # --- NEW: Store the results of this fold ---
    all_y_test.append(y_test.values)
    all_y_pred.append(y_pred)
    # ---
    
    print(f"Fold {fold_counter} Test MAE: {score:.4f}")
    print(f"Actual: {y_test.values.flatten()}, Predicted: {y_pred.flatten().round(4)}")
    
    fold_counter += 1

# --- 4. Final Results and Metrics ---
# Convert lists of predictions into single numpy arrays
all_y_test = np.vstack(all_y_test)
all_y_pred = np.vstack(all_y_pred)

# Calculate overall metrics
final_mae = mean_absolute_error(all_y_test, all_y_pred)
final_rmse = np.sqrt(mean_squared_error(all_y_test, all_y_pred))
final_r2 = r2_score(all_y_test, all_y_pred)


print("\n-----------------------------------------")
print("Cross-Validation Finished.")
print(f"Overall Mean Absolute Error (MAE): {final_mae:.4f}")
print(f"Overall Root Mean Squared Error (RMSE): {final_rmse:.4f}")
print(f"Overall R-squared (R²): {final_r2:.4f}")
print("-----------------------------------------")


# --- 5. Visualizations (Illustrations) ---

# Plot 1: Actual vs. Predicted for each output variable
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Model Performance: Actual vs. Predicted Values', fontsize=16)

# Plot for the first output variable
axes[0].scatter(all_y_test[:, 0], all_y_pred[:, 0], alpha=0.7, edgecolors='k')
axes[0].plot([all_y_test[:, 0].min(), all_y_test[:, 0].max()], [all_y_test[:, 0].min(), all_y_test[:, 0].max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Values', fontsize=12)
axes[0].set_ylabel('Predicted Values', fontsize=12)
axes[0].set_title(OUTPUT_COLS[0], fontsize=14)
axes[0].grid(True)

# Plot for the second output variable
axes[1].scatter(all_y_test[:, 1], all_y_pred[:, 1], alpha=0.7, edgecolors='k')
axes[1].plot([all_y_test[:, 1].min(), all_y_test[:, 1].max()], [all_y_test[:, 1].min(), all_y_test[:, 1].max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Values', fontsize=12)
axes[1].set_ylabel('Predicted Values', fontsize=12)
axes[1].set_title(OUTPUT_COLS[1], fontsize=14)
axes[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Plot 2: Histogram of Prediction Errors (Residuals)
residuals = all_y_test - all_y_pred
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(residuals[:, 0], kde=True, bins=10)
plt.title(f'Prediction Error Distribution for {OUTPUT_COLS[0]}', fontsize=14)
plt.xlabel('Error (Actual - Predicted)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1, 2, 2)
sns.histplot(residuals[:, 1], kde=True, bins=10)
plt.title(f'Prediction Error Distribution for {OUTPUT_COLS[1]}', fontsize=14)
plt.xlabel('Error (Actual - Predicted)', fontsize=12)

plt.suptitle('Residuals Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show all the plots
plt.show()
