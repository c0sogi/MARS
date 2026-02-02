import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config

# =============================================================================
# Scaler Persistence (No Pickle)
# =============================================================================


def save_scaler(scaler, save_dir):
    """
    Saves StandardScaler mean and scale parameters to a .npz file.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "scaler_params.npz")
    np.savez(file_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"Scaler parameters saved to {file_path}")


def load_scaler(save_dir):
    """
    Loads StandardScaler parameters from a .npz file and reconstructs the scaler.
    """
    file_path = os.path.join(save_dir, "scaler_params.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scaler file not found at {file_path}")

    data = np.load(file_path)
    scaler = StandardScaler()
    scaler.mean_ = data["mean"]
    scaler.scale_ = data["scale"]
    scaler.var_ = data["scale"] ** 2
    # n_samples_seen_ is not strictly required for transform, but good for completeness
    # We'll leave it undefined as it doesn't affect the transformation logic
    print(f"Scaler parameters loaded from {file_path}")
    return scaler


# =============================================================================
# Dataset Class
# =============================================================================


class GNSSSequenceDataset(Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        target_cols=None,
        mode="train",
        scaler=None,
        scaler_dir=Config.WORKING_DIR,
    ):
        """
        Args:
            df (pd.DataFrame): The dataframe containing features and metadata.
            feature_cols (list): List of input feature column names.
            target_cols (list, optional): List of target column names. None for inference.
            mode (str): 'train', 'val', or 'test'.
            scaler (StandardScaler, optional): Pre-fitted scaler.
            scaler_dir (str): Directory to save/load scaler params.
        """
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.mode = mode

        # Group data by drive_id and phone_name to form sequences
        # We sort by UnixTimeMillis to ensure temporal order
        self.sequences = []
        self.metadata = []

        # Grouping
        grouped = df.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group in grouped:
            group = group.sort_values("UnixTimeMillis")
            self.sequences.append(group)
            self.metadata.append(
                {
                    "drive_id": drive_id,
                    "phone_name": phone_name,
                    "timestamps": group["UnixTimeMillis"].values,
                }
            )

        # Handle Scaling
        if self.mode == "train":
            if scaler is None:
                print("Fitting scaler on training data...")
                self.scaler = StandardScaler()
                # Fit on the flattened dataframe
                self.scaler.fit(df[self.feature_cols])
                save_scaler(self.scaler, scaler_dir)
            else:
                self.scaler = scaler
        else:
            # For val/test, try to load if not provided
            if scaler is None:
                try:
                    self.scaler = load_scaler(scaler_dir)
                except FileNotFoundError:
                    print(
                        "Warning: Scaler not found for validation/test. Features will NOT be normalized."
                    )
                    self.scaler = None
            else:
                self.scaler = scaler

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df_seq = self.sequences[idx]
        meta = self.metadata[idx]

        # Extract features
        X = df_seq[self.feature_cols].values.astype(np.float32)

        # Apply normalization
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Transpose to (Channels, Length) for Conv1d compatibility
        X = X.transpose(1, 0)

        # Convert to tensor
        X_tensor = torch.from_numpy(X)

        # Extract targets if available
        y_tensor = torch.tensor([])
        if self.target_cols is not None:
            # Check if all target columns exist in dataframe
            if all(col in df_seq.columns for col in self.target_cols):
                y = df_seq[self.target_cols].values.astype(np.float32)
                # Targets shape: (Length, Output_Dim) - standard for regression losses
                y_tensor = torch.from_numpy(y)

        return X_tensor, y_tensor, meta


# =============================================================================
# Collate Function
# =============================================================================


def collate_padded_sequences(batch):
    """
    Pads sequences to the length of the longest sequence in the batch.
    Returns:
        padded_features: (Batch, Channels, Max_Length)
        padded_targets: (Batch, Max_Length, Output_Dim) or None
        masks: (Batch, Max_Length) - Boolean mask (True = valid, False = padding)
        metadata_list: List of metadata dictionaries
    """
    # Unpack batch
    features_list, targets_list, meta_list = zip(*batch)

    # Determine max length in this batch
    lengths = [f.shape[1] for f in features_list]
    max_len = max(lengths)

    batch_size = len(features_list)
    n_channels = features_list[0].shape[0]

    # Prepare padded tensors
    # Initialize with zeros
    padded_features = torch.zeros(
        (batch_size, n_channels, max_len), dtype=torch.float32
    )
    masks = torch.zeros((batch_size, max_len), dtype=torch.bool)

    has_targets = targets_list[0].numel() > 0
    padded_targets = None
    if has_targets:
        output_dim = targets_list[0].shape[1]
        padded_targets = torch.zeros(
            (batch_size, max_len, output_dim), dtype=torch.float32
        )

    # Fill tensors
    for i, length in enumerate(lengths):
        # Fill features
        padded_features[i, :, :length] = features_list[i]

        # Fill mask (True for valid data)
        masks[i, :length] = True

        # Fill targets
        if has_targets:
            padded_targets[i, :length, :] = targets_list[i]

    return padded_features, padded_targets, masks, list(meta_list)
