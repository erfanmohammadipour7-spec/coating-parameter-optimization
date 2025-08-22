# train.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib # For saving the scaler

# Import from your other custom files
from data_utils import augment_data_with_noise, CustomDataset
from model import MLP
from plotting_utils import plot_2d_results, plot_3d_surface

# --- 1. Configuration ---
# Use the corrected column names from your CSV file
FILE_PATH = 'data.csv'

INPUT_COLS = ['Current Density (mA/cm²)', 'Voltage (V)'] # <-- CORRECTED
OUTPUT_COLS = ['Porosity (%)', 'Pore Size (µm)']

# Augmentation Settings
NOISE_LEVEL = 0.05
AUGMENTATION_FACTOR = 30

# Model & Training Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 300
BATCH_SIZE = 8
HIDDEN_NEURONS = 16
DROPOUT_RATE = 0.2

# --- 2. Load Original Data ---
try:
    original_df = pd.read_csv(FILE_PATH, encoding='latin-1') # Using utf-8 is safer
except FileNotFoundError:
    print(f"Error: The data file '{FILE_PATH}' was not found.")
    print("Please make sure your data file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")

    exit()

print(f"Loaded {len(original_df)} original samples from '{FILE_PATH}'.")
print("Columns found in CSV file:", original_df.columns) 
X = original_df[INPUT_COLS]
y = original_df[OUTPUT_COLS]

# --- 3. Leave-One-Out Cross-Validation (LOOCV) ---
loo = LeaveOneOut()
all_scores = []
fold_counter = 1
all_y_test = []
all_y_pred = []

print("\nStarting Leave-One-Out Cross-Validation...")

for train_index, test_index in loo.split(X):
    print(f"\n--- Fold {fold_counter}/{len(original_df)} ---")
    
    # Split data into training and testing for this fold
    X_train_orig, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_orig, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Augment ONLY the training data for this fold
    df_to_augment = pd.concat([X_train_orig, y_train_orig], axis=1)
    augmented_part = augment_data_with_noise(df_to_augment, INPUT_COLS, NOISE_LEVEL, AUGMENTATION_FACTOR)
    
    # Combine original training data with the new augmented data
    X_train_aug = pd.concat([X_train_orig, augmented_part[INPUT_COLS]], ignore_index=True)
    y_train_aug = pd.concat([y_train_orig, augmented_part[OUTPUT_COLS]], ignore_index=True)
    print(f"Training on {len(X_train_orig)} original + {len(augmented_part)} augmented samples = {len(X_train_aug)} total.")

    # Feature Scaling: Fit on original training data, transform both train and test
    scaler = StandardScaler().fit(X_train_orig)
    X_train_scaled = scaler.transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    # Create PyTorch Datasets and DataLoader
    train_dataset = CustomDataset(X_train_scaled, y_train_aug.values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize the Model, Loss Function, and Optimizer for this fold
    model = MLP(input_size=len(INPUT_COLS), output_size=len(OUTPUT_COLS), hidden_neurons=HIDDEN_NEURONS, dropout_rate=DROPOUT_RATE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Model Training Loop
    model.train()
    for epoch in range(EPOCHS):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- Evaluation for this fold ---
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()

    # Store results for final analysis
    all_y_test.append(y_test.values)
    all_y_pred.append(y_pred)
    
    # Print fold-specific results
    score = mean_absolute_error(y_test, y_pred)
    all_scores.append(score)
    print(f"Fold {fold_counter} Test MAE: {score:.4f}")
    print(f"Actual: {y_test.values.flatten()}, Predicted: {y_pred.flatten().round(4)}")
    
    fold_counter += 1

# --- 4. Final Results and Metrics from Cross-Validation ---
# Combine results from all folds into single arrays for overall metric calculation
all_y_test = np.vstack(all_y_test)
all_y_pred = np.vstack(all_y_pred)

final_mae = mean_absolute_error(all_y_test, all_y_pred)
final_rmse = np.sqrt(mean_squared_error(all_y_test, all_y_pred))
final_r2 = r2_score(all_y_test, all_y_pred)

print("\n-----------------------------------------")
print("Cross-Validation Finished.")
print(f"Overall Mean Absolute Error (MAE): {final_mae:.4f}")
print(f"Overall Root Mean Squared Error (RMSE): {final_rmse:.4f}")
print(f"Overall R-squared (R²): {final_r2:.4f}")
print("-----------------------------------------")


# --- 5. Train Final Model on ALL Data for Visualization and Optimization ---
# A single model trained on all data gives a smoother surface for plotting and a final state to save.
print("\nTraining a final model on all data for visualization...")

# Augment the full original dataset
full_df_to_augment = pd.concat([X, y], axis=1)
full_augmented_part = augment_data_with_noise(full_df_to_augment, INPUT_COLS, NOISE_LEVEL, AUGMENTATION_FACTOR)

X_full_aug = pd.concat([X, full_augmented_part[INPUT_COLS]], ignore_index=True)
y_full_aug = pd.concat([y, full_augmented_part[OUTPUT_COLS]], ignore_index=True)

# Fit the final scaler ONLY on the original, non-augmented data
final_scaler = StandardScaler().fit(X) 
X_full_scaled = final_scaler.transform(X_full_aug)

# Create the final DataLoader
final_dataset = CustomDataset(X_full_scaled, y_full_aug.values)
final_loader = DataLoader(final_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate and train the final model
final_model = MLP(input_size=len(INPUT_COLS), output_size=len(OUTPUT_COLS), hidden_neurons=HIDDEN_NEURONS, dropout_rate=DROPOUT_RATE)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

final_model.train()
for epoch in range(EPOCHS):
    for features, labels in final_loader:
        optimizer.zero_grad()
        outputs = final_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Final model training complete.")

# --- 6. Generate All Visualizations ---
# Call the plotting functions from the separate utility file

# Generate 2D plots from the cross-validation results
plot_2d_results(all_y_test, all_y_pred, OUTPUT_COLS)

# Generate 3D surface plot using the final trained model
plot_3d_surface(final_model, original_df, INPUT_COLS, OUTPUT_COLS, final_scaler)

# Show all generated plot windows
plt.show()

# --- 7. Save the Final Model and Scaler for Optimization ---
print("\n-----------------------------------------")
print("Saving final model and scaler to disk...")

# Save the model's learned weights
torch.save(final_model.state_dict(), 'final_model.pth')

# Save the scaler object using joblib
joblib.dump(final_scaler, 'final_scaler.pkl')

print("Files 'final_model.pth' and 'final_scaler.pkl' have been saved.")
print("You can now run the 'optimize_inputs.py' script.")
print("-----------------------------------------")
