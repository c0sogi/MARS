import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the NFL Contact Detection Task.

    Splits the flattened numerical feature vector into:
    1. Kinematic Features (Tracking data + Derived physics metrics)
    2. Visual Features (Helmet box metrics)

    This separation allows the Dual-Stream TD-SRV-Net to process physical dynamics
    and visual cues through distinct architectural branches.
    """

    def __init__(self, X_num, X_cat, y=None):
        """
        Args:
            X_num (np.ndarray): Scaled numerical features (N_samples, N_features).
            X_cat (np.ndarray): Encoded categorical features (N_samples, N_cat_features).
            y (np.ndarray, optional): Binary targets (N_samples,). Defaults to None.
        """
        self.y = torch.FloatTensor(y) if y is not None else None

        # --- Feature Separation Logic ---
        # We need to de-interleave the flattened X_num based on the order defined in NFLDataLoader.
        # Order per lag: [P1_Kin, P1_Vis, P2_Kin, P2_Vis, Derived]

        n_kin = len(Config.KINEMATIC_FEATURES)
        n_vis = len(Config.VISUAL_FEATURES)
        # Derived features defined in DataLoader: rel_dist, rel_speed, rel_accel, closing_speed
        n_derived = 4

        features_per_lag = n_kin + n_vis + n_kin + n_vis + n_derived
        window_size = Config.WINDOW_SIZE

        # Verify shape matches expectation
        expected_cols = window_size * features_per_lag
        if X_num.shape[1] != expected_cols:
            raise ValueError(
                f"Expected {expected_cols} columns in X_num, got {X_num.shape[1]}"
            )

        kinematic_indices = []
        visual_indices = []

        for t in range(window_size):
            offset = t * features_per_lag

            # 1. P1 Kinematic
            start = offset
            end = start + n_kin
            kinematic_indices.extend(range(start, end))

            # 2. P1 Visual
            start = end
            end = start + n_vis
            visual_indices.extend(range(start, end))

            # 3. P2 Kinematic
            start = end
            end = start + n_kin
            kinematic_indices.extend(range(start, end))

            # 4. P2 Visual
            start = end
            end = start + n_vis
            visual_indices.extend(range(start, end))

            # 5. Derived (Kinematic context)
            start = end
            end = start + n_derived
            kinematic_indices.extend(range(start, end))

        # Slice the numpy arrays
        X_kin_np = X_num[:, kinematic_indices]
        X_vis_np = X_num[:, visual_indices]

        # Convert to Tensors
        self.X_kin = torch.FloatTensor(X_kin_np)
        self.X_vis = torch.FloatTensor(X_vis_np)
        self.X_cat = torch.LongTensor(X_cat)

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: ((x_kin, x_vis, x_cat), y)

            x_kin: (Window_Size * 36,) - Flattened kinematic features
            x_vis: (Window_Size * 10,) - Flattened visual features
            x_cat: (4,) - Categorical indices [pos1, team1, pos2, team2]
            y: (1,) - Target label (or -1.0 if test)
        """
        x_kin = self.X_kin[idx]
        x_vis = self.X_vis[idx]
        x_cat = self.X_cat[idx]

        if self.y is not None:
            target = self.y[idx]
        else:
            # Placeholder for test set
            target = torch.tensor(-1.0)

        return (x_kin, x_vis, x_cat), target
