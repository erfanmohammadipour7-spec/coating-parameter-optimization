
---

# Predicting Material Properties with a Neural Network

This project demonstrates how to build, train, and evaluate a Multi-Layer Perceptron (MLP) neural network using PyTorch. The primary goal is to predict material properties (`Porosity` and `Pore Size`) based on manufacturing process parameters (`Current Density` and `Voltage`).

Given the small size of the initial dataset (9 samples), the project employs key techniques like **data augmentation** and **Leave-One-Out Cross-Validation (LOOCV)** to create a robust model and provide a reliable performance evaluation.

The final output includes performance metrics (MAE, RMSE, R²) and rich 2D and 3D visualizations to interpret the model's behavior.

## Key Features

-   **Neural Network Model**: A simple and effective MLP built with PyTorch.
-   **Data Augmentation**: Artificially expands the training dataset by adding Gaussian noise, helping the model generalize better from limited data.
-   **Robust Validation**: Uses Leave-One-Out Cross-Validation (LOOCV), an exhaustive method ideal for very small datasets.
-   **Comprehensive Evaluation**: Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) to assess model performance.
-   **Rich Visualizations**:
    -   2D plots of **Actual vs. Predicted** values to check prediction accuracy.
    -   2D histograms of **Prediction Errors (Residuals)** to analyze model bias.
    -   3D **Surface Plots** to visualize the learned relationship between the two input parameters and each output property.
-   **Modular Code**: The project is organized into separate files for data utilities, model architecture, plotting, and training, making it easy to understand and maintain.

## Project Structure

```
.
├── data.csv            # The dataset containing input and output variables
├── data_utils.py       # Contains functions for data augmentation and the custom PyTorch Dataset
├── model.py            # Defines the MLP neural network architecture
├── plotting_utils.py   # Contains all functions for generating 2D and 3D plots
└── train.py            # The main script to load data, run the training loop, and generate results
```

## How It Works

The workflow is managed by the `train.py` script and can be broken down into the following steps:

1.  **Configuration**: All key parameters, such as file paths, column names, and model hyperparameters (learning rate, epochs, etc.), are defined at the top of `train.py`.
2.  **Data Loading**: The script loads the `data.csv` file into a pandas DataFrame.
3.  **Leave-One-Out Cross-Validation (LOOCV)**: To get a reliable measure of performance with only 9 data points, the model is trained and evaluated 9 times. In each "fold," 8 samples are used for training and the remaining 1 is used for testing.
4.  **Data Augmentation**: Within each LOOCV fold, the 8 training samples are augmented. The `augment_data_with_noise` function creates multiple new training samples from each original one by adding a small amount of random noise. This prevents the model from overfitting and helps it learn the underlying patterns more effectively.
5.  **Training**: For each fold, a new MLP model is instantiated. The augmented training data is used to train the model for a fixed number of epochs.
6.  **Evaluation**: The trained model's performance is evaluated on the single, unseen test sample for that fold. The predictions and actual values are stored.
7.  **Final Metrics**: After all folds are complete, the collected predictions are compared against the actual values to calculate the overall MAE, RMSE, and R² scores, which represent the model's generalized performance.
8.  **Visualization**:
    -   A final model is trained on the *entire* augmented dataset to create a smooth surface for plotting.
    -   The `plotting_utils.py` script is called to generate the 2D and 3D plots, providing a visual summary of the model's performance.

## Getting Started

Follow these steps to set up and run the project.

### 1. Prerequisites

Make sure you have Python 3 installed. You will also need the following libraries:

-   `torch`
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`

### 2. Installation

You can install all the required libraries using pip. It's recommended to create a virtual environment first.

```bash
# Create a file named requirements.txt with the following content:
# pandas
# numpy
# torch
# scikit-learn
# matplotlib
# seaborn

# Install the libraries
pip install -r requirements.txt
```

### 3. Running the Script

With the libraries installed and all the `.py` files in the same directory, run the main training script from your terminal:

```bash
python train.py
```

The script will print the progress of the cross-validation for each fold, followed by the final performance metrics. Finally, plot windows will appear showing the visualizations.

## How to Use Your Own Data

This project is designed to be easily adaptable. To use your own dataset, follow these steps:

1.  **Prepare your CSV file**: Ensure it has clear column headers.
2.  **Place it in the project directory**.

## Results
Add some explanation......


![images .....](https://github.com/erfanmohammadipour7-spec/coating-parameter-optimization/blob/main/Images/Figure_22.png)
