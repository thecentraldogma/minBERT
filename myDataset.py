import torch
from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_file, feature_cols, target_col, dtype=torch.float32):
        """
        Args:
            csv_file (str): Path to the CSV file
            feature_cols (list): List of column names (features)
            target_col (str): Name of the target column
            transform (callable, optional): Optional transform to apply to features
            dtype (torch.dtype): dtype for tensors (default: float32)
        """
        self.data = pd.read_csv(csv_file)
        self.features = self.data[feature_cols].values
        self.targets = self.data[target_col].values
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=self.dtype)
        y = torch.tensor(self.targets[idx], dtype=self.dtype)


        return X, y

