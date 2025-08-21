# data_utils.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def augment_data_with_noise(df, input_cols, noise_level=0.05, augmentation_factor=10):
    """
    Augments a DataFrame by adding Gaussian noise to specified input columns.

    Args:
        df (pd.DataFrame): The original DataFrame to augment.
        input_cols (list): A list of column names to which noise will be added.
        noise_level (float): The standard deviation of the noise as a fraction
                             of the data's standard deviation.
        augmentation_factor (int): How many new samples to generate for each
                                   original sample.

    Returns:
        pandas.DataFrame: A new DataFrame containing only the augmented samples.
    """
    if df.empty:
        return pd.DataFrame()

    input_std = df[input_cols].std()
    augmented_data_list = []

    for _, original_row in df.iterrows():
        for _ in range(augmentation_factor):
            new_row = original_row.copy()
            for col in input_cols:
                noise = np.random.normal(0, input_std[col] * noise_level)
                new_row[col] += noise
            augmented_data_list.append(new_row)

    return pd.DataFrame(augmented_data_list)


class CustomDataset(Dataset):
    """Custom PyTorch Dataset for tabular data."""
    def __init__(self, features, labels):
        # Convert data to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        # Return the total number of samples
        return len(self.features)

    def __getitem__(self, idx):
        # Retrieve a sample at a given index
        return self.features[idx], self.labels[idx]
