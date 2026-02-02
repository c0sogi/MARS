import torch
from torch.utils.data import Dataset
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.features import generate_features


class NFLContactDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, debug=False):
        """
        PyTorch Dataset for NFL Contact Detection.

        Loads precomputed features, applies normalization (StandardScaler),
        and prepares tensors for the Dual-Stream architecture.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): If True, loads raw features from cache.
            debug (bool): If True, uses a small subset of data.
        """
        self.split = split

        # 1. Load Raw Features
        # Relies on library.features for heavy lifting and caching of raw data
        X_kin, X_vis, X_cat, y, ids = generate_features(
            split=split, load_cached_data=load_cached_data, debug=debug
        )

        # 2. Normalization Strategy
        # We apply StandardScaler to all continuous features (Kinematic + Visual).
        # We fit on Train, and transform Validation/Test.

        scaler_path = os.path.join(Config.WORKING_DIR, "scaler.joblib")

        # Concatenate continuous streams for unified scaling management
        # X_kin: [N, Kin_Dim], X_vis: [N, Vis_Dim]
        X_cont = np.hstack([X_kin, X_vis])

        if split == "train":
            # Fit on training data
            self.scaler = StandardScaler()
            self.scaler.fit(X_cont)

            # Save scaler for consistency across splits
            joblib.dump(self.scaler, scaler_path)

            # Transform training data
            X_cont = self.scaler.transform(X_cont)

        else:
            # Load existing scaler
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. "
                    "You must initialize the dataset with split='train' first to fit the scaler."
                )

            self.scaler = joblib.load(scaler_path)

            # Transform validation/test data
            X_cont = self.scaler.transform(X_cont)

        # 3. Convert to Tensors and Split Streams
        # Split back into Kinematic and Visual streams based on original dimensions
        kin_dim = X_kin.shape[1]

        # Use float32 for model weights
        self.X_kin = torch.FloatTensor(X_cont[:, :kin_dim])
        self.X_vis = torch.FloatTensor(X_cont[:, kin_dim:])

        # Categorical features are indices (LongTensor)
        self.X_cat = torch.LongTensor(X_cat)

        # Targets
        self.y = torch.FloatTensor(y)

        # Metadata
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """
        Returns a dictionary of features and target for a single sample.
        """
        return {
            "kinematic": self.X_kin[idx],
            "visual": self.X_vis[idx],
            "categorical": self.X_cat[idx],
            "target": self.y[idx],
            "contact_id": self.ids[idx],
        }
