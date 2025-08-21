# plotting_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def plot_2d_results(all_y_test, all_y_pred, output_cols):
    """
    Generates and displays 2D plots for model evaluation.
    - Actual vs. Predicted scatter plots
    - Residuals (prediction error) histograms

    Args:
        all_y_test (np.ndarray): Array of all true target values from cross-validation.
        all_y_pred (np.ndarray): Array of all predicted values from cross-validation.
        output_cols (list): List of names for the output columns.
    """
    print("Generating 2D evaluation plots...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Actual vs. Predicted for each output variable
    fig, axes = plt.subplots(1, len(output_cols), figsize=(8 * len(output_cols), 6))
    fig.suptitle('Model Performance: Actual vs. Predicted Values', fontsize=16)
    
    # Ensure axes is always a list for consistent indexing
    if len(output_cols) == 1:
        axes = [axes]

    for i, col_name in enumerate(output_cols):
        axes[i].scatter(all_y_test[:, i], all_y_pred[:, i], alpha=0.7, edgecolors='k')
        axes[i].plot([all_y_test[:, i].min(), all_y_test[:, i].max()], 
                     [all_y_test[:, i].min(), all_y_test[:, i].max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual Values', fontsize=12)
        axes[i].set_ylabel('Predicted Values', fontsize=12)
        axes[i].set_title(col_name, fontsize=14)
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot 2: Histogram of Prediction Errors (Residuals)
    residuals = all_y_test - all_y_pred
    plt.figure(figsize=(8 * len(output_cols), 6))
    
    for i, col_name in enumerate(output_cols):
        plt.subplot(1, len(output_cols), i + 1)
        sns.histplot(residuals[:, i], kde=True, bins=10)
        plt.title(f'Prediction Error Distribution for {col_name}', fontsize=14)
        plt.xlabel('Error (Actual - Predicted)', fontsize=12)
        if i == 0:
            plt.ylabel('Frequency', fontsize=12)

    plt.suptitle('Residuals Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_3d_surface(final_model, original_df, input_cols, output_cols, scaler):
    """
    Generates and displays a 3D surface plot of the model's predictions.

    Args:
        final_model (torch.nn.Module): The fully trained MLP model.
        original_df (pd.DataFrame): The original, unaugmented dataframe.
        input_cols (list): List of names for the input columns (must be 2).
        output_cols (list): List of names for the output columns.
        scaler (StandardScaler): The scaler fitted on the original training data.
    """
    if len(input_cols) != 2:
        print("Cannot create 3D plot: exactly two input variables are required.")
        return
        
    print("Generating 3D visualization...")
    X = original_df[input_cols]
    y = original_df[output_cols]

    # 1. Create a grid of points to predict on
    x_min, x_max = X[input_cols[0]].min() - 1, X[input_cols[0]].max() + 1
    y_min, y_max = X[input_cols[1]].min() - 1, X[input_cols[1]].max() + 1
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    xx, yy = np.meshgrid(x_range, y_range)

    # 2. Prepare grid points for the model
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    grid_points_tensor = torch.tensor(grid_points_scaled, dtype=torch.float32)

    # 3. Get model predictions for the entire grid
    final_model.eval()
    with torch.no_grad():
        zz_pred_tensor = final_model(grid_points_tensor)
        zz_pred = zz_pred_tensor.numpy()

    # Reshape predictions to match the grid shape for each output
    zz_outputs = [zz_pred[:, i].reshape(xx.shape) for i in range(len(output_cols))]
    
    # 4. Plotting
    fig = plt.figure(figsize=(9 * len(output_cols), 8))
    fig.suptitle('3D View of Model Predictions vs. Actual Data', fontsize=16)

    for i, col_name in enumerate(output_cols):
        ax = fig.add_subplot(1, len(output_cols), i + 1, projection='3d')
        # Scatter plot of the original data points
        ax.scatter(X[input_cols[0]], X[input_cols[1]], y[col_name], 
                   c='red', marker='o', label='Actual Data Points', s=50, depthshade=True)
        # Surface plot of the model's predictions
        ax.plot_surface(xx, yy, zz_outputs[i], alpha=0.6, cmap='viridis', rstride=1, cstride=1, edgecolor='none')
        ax.set_xlabel(input_cols[0], fontsize=12)
        ax.set_ylabel(input_cols[1], fontsize=12)
        ax.set_zlabel(col_name, fontsize=12)
        ax.set_title(f'Predicted Surface for {col_name}', fontsize=14)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
