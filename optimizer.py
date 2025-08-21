# optimizer.py

import torch
import torch.optim as optim
import numpy as np

def objective_function(outputs):
    """
    Defines the function to be minimized.
    This is the place to define the relationship between your outputs.

    Current implementation: Minimize the sum of the two outputs.

    Args:
        outputs (torch.Tensor): The model's predictions, shape (n_samples, n_outputs).

    Returns:
        torch.Tensor: A single value for each sample, shape (n_samples,).
    """
    # Example: Simple sum. Porosity + Pore Size
    return outputs[:, 0] + outputs[:, 1]

    # Example: Weighted sum. Give more importance to minimizing the first output.
    # return 1.5 * outputs[:, 0] + 0.5 * outputs[:, 1]


def find_minimum_grid_search(model, scaler, original_df, input_cols, output_cols):
    """
    Finds the minimum of the objective function using a dense grid search.

    Args:
        model (torch.nn.Module): The trained final model.
        scaler (StandardScaler): The scaler fitted on the original data.
        original_df (pd.DataFrame): The original dataframe to define input ranges.
        input_cols (list): List of input column names.
        output_cols (list): List of output column names.

    Returns:
        dict: A dictionary containing the optimal inputs and predicted outputs.
    """
    print("\n--- Starting Grid Search for Optimal Inputs ---")

    # 1. Create a dense grid of input points
    num_points = 200 # Increase for higher precision
    x_min, x_max = original_df[input_cols[0]].min(), original_df[input_cols[0]].max()
    y_min, y_max = original_df[input_cols[1]].min(), original_df[input_cols[1]].max()
    x_range = np.linspace(x_min, x_max, num_points)
    y_range = np.linspace(y_min, y_max, num_points)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 2. Scale inputs and predict with the model
    grid_points_scaled = scaler.transform(grid_points)
    grid_tensor = torch.tensor(grid_points_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(grid_tensor)

    # 3. Calculate the objective function for all points
    objective_values = objective_function(predictions).numpy()

    # 4. Find the minimum
    min_index = np.argmin(objective_values)
    min_objective_value = objective_values[min_index]

    # Get the corresponding inputs and outputs
    optimal_inputs = grid_points[min_index]
    optimal_outputs = predictions[min_index].numpy()

    result = {
        'optimal_inputs': dict(zip(input_cols, optimal_inputs)),
        'predicted_outputs': dict(zip(output_cols, optimal_outputs)),
        'min_objective_value': min_objective_value
    }
    return result


def find_minimum_gradient_descent(model, scaler, output_cols, n_starts=10, steps=200, lr=0.1):
    """
    Finds the minimum of the objective function using gradient descent on the inputs.

    Args:
        model (torch.nn.Module): The trained final model.
        scaler (StandardScaler): The scaler fitted on the original data.
        output_cols (list): List of output column names.
        n_starts (int): Number of random starting points to avoid local minima.
        steps (int): Number of optimization steps.
        lr (float): Learning rate for the optimizer.

    Returns:
        dict: A dictionary containing the optimal inputs and predicted outputs.
    """
    print("\n--- Starting Gradient Descent for Optimal Inputs ---")

    best_overall_objective = float('inf')
    best_inputs = None

    # The search is performed on the scaled data, as the model was trained on it
    for i in range(n_starts):
        # --- THIS IS THE CORRECTED PART ---
        # 1. Create the initial data and clamp it. This is just data, not a graph node.
        initial_data = torch.randn(1, scaler.n_features_in_).clamp(-2, 2)
        # 2. Now, create a new leaf tensor from this data that requires a gradient.
        inputs = torch.tensor(initial_data, requires_grad=True)
        # --- END OF CORRECTION ---

        # We optimize the inputs, not the model weights
        optimizer = optim.Adam([inputs], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()

            # Get model output for the current inputs
            outputs = model(inputs)

            # Calculate the objective
            objective = objective_function(outputs)

            # Backpropagate to get gradients of objective w.r.t. inputs
            objective.backward()

            # Update the inputs to minimize the objective
            optimizer.step()

        # Check if this run found a better minimum
        final_objective = objective.item()
        if final_objective < best_overall_objective:
            best_overall_objective = final_objective
            best_inputs = inputs.detach()

    # Unscale the final best inputs to get the original values
    optimal_inputs_scaled = best_inputs.numpy()
    optimal_inputs_unscaled = scaler.inverse_transform(optimal_inputs_scaled)

    # Get the final predicted outputs
    model.eval()
    with torch.no_grad():
        optimal_outputs = model(best_inputs).numpy()

    result = {
        'optimal_inputs': dict(zip(scaler.get_feature_names_out(), optimal_inputs_unscaled[0])),
        'predicted_outputs': dict(zip(output_cols, optimal_outputs[0])),
        'min_objective_value': best_overall_objective
    }
    return result
