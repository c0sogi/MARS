import torch
from torch.utils.data import Dataset
import numpy as np
import library.config as config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the NR-PIRV-Net architecture.

    Handles the dual-stream input (Kinematic and Visual) and implements
    Input Noise Injection for structural regularization during training.
    """

    def __init__(self, df, split="train"):
        """
        Args:
            df (pd.DataFrame): The processed dataframe containing features and metadata.
            split (str): One of 'train', 'validation', 'test'. Determines augmentation logic.
        """
        self.split = split

        # --- Column Identification ---
        # 1. Identify Visual Columns (Explicitly defined in config)
        # Structure: [feat_1, feat_2] for each feature in VISUAL_FEATURES
        self.vis_cols = []
        for c in config.VISUAL_FEATURES:
            self.vis_cols.append(f"{c}_1")
            self.vis_cols.append(f"{c}_2")

        # 2. Identify Metadata and Target Columns
        # We exclude these to find the Kinematic columns
        exclude_cols = set(config.META_COLUMNS) | {
            "contact_id",
            "contact",
            "nfl_player_id_1",
            "nfl_player_id_2",
        }
        exclude_cols.update(self.vis_cols)

        # 3. Identify Kinematic Columns
        # All remaining numeric columns are kinematic features (lags, relative geometry, dynamics)
        self.kin_cols = [
            c
            for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        # --- Data Storage ---
        # Convert to float32 numpy arrays for efficiency
        self.kinematic_data = df[self.kin_cols].values.astype(np.float32)
        self.visual_data = df[self.vis_cols].values.astype(np.float32)

        # Handle Targets
        if "contact" in df.columns:
            self.targets = df["contact"].values.astype(np.float32)
        else:
            # For test set inference where targets might not exist
            self.targets = np.zeros(len(df), dtype=np.float32)

        # Store Noise Sigma
        self.noise_sigma = config.NOISE_SIGMA

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve base features
        kin_feat = self.kinematic_data[idx]
        vis_feat = self.visual_data[idx]
        target = self.targets[idx]

        # --- Input Noise Injection ---
        # Only applied to Kinematic features during training
        if self.split == "train":
            noise = np.random.normal(0, self.noise_sigma, size=kin_feat.shape).astype(
                np.float32
            )
            kin_feat = kin_feat + noise

        # Convert to Tensors
        return {
            "kinematic": torch.from_numpy(kin_feat),
            "visual": torch.from_numpy(vis_feat),
            "target": torch.tensor(target, dtype=torch.float32),
        }


# Import pandas locally to handle column type checking inside __init__
import pandas as pd
