import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, input_npy, target_npy, transform=None):
        self.inputs = np.load(input_npy)
        self.targets = np.load(target_npy)

        # Add an extra dimension for channels if not already present
        if self.inputs.ndim == 2:  # Assuming shape is (num_samples, sequence_length)
            self.inputs = np.expand_dims(self.inputs, axis=1)  # New shape: (num_samples, channels, sequence_length)
        if self.targets.ndim == 2:  # Assuming shape is (num_samples, sequence_length)
            self.targets = np.expand_dims(self.targets, axis=1)  # New shape: (num_samples, channels, sequence_length)

        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ts = torch.tensor(self.inputs[index], dtype=torch.float32)
        target_ts = torch.tensor(self.targets[index], dtype=torch.float32)

        if self.transform is not None:
            augmentations = self.transform(input_ts=input_ts, target_ts=target_ts)
            input_ts = augmentations["input_ts"]
            target_ts = augmentations["target_ts"]

        return input_ts, target_ts

class SpectrogramDataset(Dataset):
    def __init__(self, input_npy, target_npy, transform=None):
        # Store paths to input and target .npy files
        self.input_path = input_npy
        self.target_path = target_npy

        # Load the shape/length of the data without loading the entire dataset into memory
        self.input_shape = np.load(input_npy, mmap_mode='r').shape
        self.target_shape = np.load(target_npy, mmap_mode='r').shape
        
        # Ensure dimensions are correct and check for possible expansions
        self.input_channels_needed = len(self.input_shape) == 3
        self.target_channels_needed = len(self.target_shape) == 3

        self.transform = transform

    def __len__(self):
        return self.input_shape[0]  # Number of samples

    def __getitem__(self, index):
        # Load the specific data sample from the .npy files using memory-mapped mode
        input_ts = np.load(self.input_path, mmap_mode='r')[index]  # Load only the required sample
        target_ts = np.load(self.target_path, mmap_mode='r')[index]

        # Add an extra dimension for channels if needed (assuming shape: (num_samples, height, width))
        if self.input_channels_needed:
            input_ts = np.expand_dims(input_ts, axis=0)  # Shape: (channels, height, width)
        if self.target_channels_needed:
            target_ts = np.expand_dims(target_ts, axis=0)  # Shape: (channels, height, width)

        # Convert inputs and targets to torch tensors
        input_ts = torch.tensor(input_ts, dtype=torch.float32)  # Shape: (channels, height, width)
        target_ts = torch.tensor(target_ts, dtype=torch.float32)  # Shape: (channels, height, width)

        # Apply transformations if provided
        if self.transform is not None:
            augmentations = self.transform(input_ts=input_ts, target_ts=target_ts)
            input_ts = augmentations["input_ts"]
            target_ts = augmentations["target_ts"]

        return input_ts, target_ts