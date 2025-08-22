# optimize_bayesian.py

import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt # <--- ADD THIS IMPORT

# Import the Bayesian Optimizer from scikit-optimize
from skopt import gp_minimize
from skopt.space import Real

# Import the model structure from model.py
from model import MLP

# --- 1. Load the Trained Model and Scaler ---

print("Loading trained model and scaler...")
try:
    scaler = joblib.load('final_scaler.pkl')
    model = MLP(
        input_size=2,
        output_size=2,
        hidden_neurons=16,
        dropout_rate=0.2
    )
    model.load_state_dict(torch.load('final_model.pth'))
    model.eval()
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please run train.py first.")
    exit()

# --- 2. Define the Objective Function for the Optimizer ---

INPUT_COLS = ['Current Density (mA/cm²)', 'Voltage (V)']
OUTPUT_COLS = ['Porosity (%)', 'Pore Size (µm)']

def objective_function(params):
    current_density, voltage = params
    input_data = np.array([[current_density, voltage]])
    input_df = pd.DataFrame(input_data, columns=INPUT_COLS)
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).numpy()[0]
    pore_size = prediction[1]
    return pore_size

# --- 3. Define the Search Space (Constraints) ---

search_space = [
    Real(50, 400, name='Current Density (mA/cm²)'),
    Real(250, 600, name='Voltage (V)')
]

# --- 4. Run the Bayesian Optimization ---

print("\nStarting Bayesian Optimization...")
print("Objective: MINIMIZE Pore Size (µm)")

result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=75,
    random_state=42
)

print("Optimization Complete.")

# --- 5. Display the Optimal Result ---

best_params = result.x
min_score = result.fun

final_prediction_porosity, final_prediction_pore_size = model(
    torch.tensor(scaler.transform(
        pd.DataFrame([best_params], columns=INPUT_COLS)
    ), dtype=torch.float32)
).detach().numpy()[0]

print("\n-----------------------------------------")
print(f"Minimum objective score (Pore Size) found: {min_score:.4f}")
print("\nBest Input Conditions to achieve this minimum:")
for i, dim in enumerate(search_space):
    print(f"  - {dim.name}: {best_params[i]:.2f}")

print("\nPredicted Outcome at these conditions:")
print(f"  - {OUTPUT_COLS[0]}: {final_prediction_porosity:.4f}")
print(f"  - {OUTPUT_COLS[1]}: {final_prediction_pore_size:.4f} (This is the minimized value)")
print("-----------------------------------------")


# --- 6. Generate Visualization Plots ---

print("\nGenerating visualization plots...")

# Get all the input points that the optimizer tested
all_tested_inputs = np.array(result.x_iters)

# To get the corresponding outputs, we need to run all tested inputs through the model
all_inputs_df = pd.DataFrame(all_tested_inputs, columns=INPUT_COLS)
all_inputs_scaled = scaler.transform(all_inputs_df)
all_inputs_tensor = torch.tensor(all_inputs_scaled, dtype=torch.float32)

with torch.no_grad():
    all_predicted_outputs = model(all_inputs_tensor).numpy()

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- PLOT 1: INPUT SEARCH SPACE ---
plt.figure(figsize=(10, 8))
# Plot all the blue points that were tested
plt.scatter(
    all_tested_inputs[:, 0], # All Current Density values
    all_tested_inputs[:, 1], # All Voltage values
    c='blue',
    alpha=0.6,
    label='Tested Points'
)
# Plot the final red optimal point
plt.scatter(
    best_params[0], # Best Current Density
    best_params[1], # Best Voltage
    c='red',
    s=150, # Make it bigger
    edgecolors='black',
    zorder=10, # Make sure it's on top
    label='Optimal Point'
)
plt.title('Bayesian Optimization Search Path (Input Space)', fontsize=16)
plt.xlabel(INPUT_COLS[0], fontsize=12)
plt.ylabel(INPUT_COLS[1], fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# --- PLOT 2: OUTPUT (OBJECTIVE) SPACE ---
plt.figure(figsize=(10, 8))
# Plot all the blue predicted outcomes
plt.scatter(
    all_predicted_outputs[:, 0], # All predicted Porosity values
    all_predicted_outputs[:, 1], # All predicted Pore Size values
    c='blue',
    alpha=0.6,
    label='Predicted Outcomes'
)
# Plot the final red optimal outcome
plt.scatter(
    final_prediction_porosity,
    final_prediction_pore_size,
    c='red',
    s=150,
    edgecolors='black',
    zorder=10,
    label='Optimal Outcome'
)
plt.title('Predicted Model Outcomes (Output Space)', fontsize=16)
plt.xlabel(OUTPUT_COLS[0], fontsize=12)
plt.ylabel(OUTPUT_COLS[1], fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Show both plots
plt.show()
