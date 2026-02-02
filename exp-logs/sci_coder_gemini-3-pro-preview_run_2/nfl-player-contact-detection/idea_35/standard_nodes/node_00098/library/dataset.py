import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.feature_engineering import prepare_features


class ContactDataset(Dataset):
    """
    PyTorch Dataset for APIRV-Net.
    Splits the input feature vector into Kinematic and Visual streams based on column names.
    """

    def __init__(self, X, y, feature_names):
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target vector.
            feature_names (list): List of feature column names corresponding to X.
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

        # Determine indices for the two streams
        self.kin_indices = []
        self.vis_indices = []

        # Visual features contain specific keywords based on feature_engineering.py
        # Kinematic features are the rest (mostly containing _t suffix or dynamics)
        vis_keywords = ["visual_", "left", "top", "width", "height"]

        for idx, col in enumerate(feature_names):
            if any(k in col for k in vis_keywords):
                self.vis_indices.append(idx)
            else:
                self.kin_indices.append(idx)

        self.kin_indices = torch.tensor(self.kin_indices, dtype=torch.long)
        self.vis_indices = torch.tensor(self.vis_indices, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Extract streams
        x_kin = self.X[idx, self.kin_indices]
        x_vis = self.X[idx, self.vis_indices]

        # Return tuple of inputs to match model forward signature
        inputs = (x_kin, x_vis)

        if self.y is not None:
            return inputs, self.y[idx]
        return inputs


def _get_feature_names(cache_path):
    """
    Helper to extract feature column names from the cached parquet file.
    Excludes metadata columns added during saving.
    """
    # Read the dataframe to get columns.
    # prepare_features saves the dataframe, so this file is guaranteed to exist if prepare_features succeeded.
    df = pd.read_parquet(cache_path)

    # Metadata columns added by prepare_features to exclude from feature set
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]

    feature_names = [c for c in df.columns if c not in meta_cols]
    return feature_names


def get_train_val_datasets(debug=False):
    """
    Generates and loads Training and Validation datasets.
    Uses monkey-patching to redirect prepare_features to validation metadata/cache.

    Args:
        debug (bool): If True, uses a subset of data.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # --- Load Training Data ---
    # prepare_features uses Config.TRAIN_META_PATH by default when train_mode=True
    X_train, y_train, _ = prepare_features(train_mode=True, debug=debug)
    train_feats = _get_feature_names(Config.CACHE_TRAIN_FEATURES)

    train_dataset = ContactDataset(X_train, y_train, train_feats)

    # --- Load Validation Data ---
    # Save original config paths
    orig_meta = Config.TRAIN_META_PATH
    orig_cache = Config.CACHE_TRAIN_FEATURES

    # Redirect Config to Validation paths
    # This forces prepare_features to process the validation set
    Config.TRAIN_META_PATH = Config.VAL_META_PATH
    Config.CACHE_TRAIN_FEATURES = Config.CACHE_VAL_FEATURES

    try:
        X_val, y_val, _ = prepare_features(train_mode=True, debug=debug)
        val_feats = _get_feature_names(Config.CACHE_VAL_FEATURES)
        val_dataset = ContactDataset(X_val, y_val, val_feats)
    finally:
        # Restore original config to prevent side effects
        Config.TRAIN_META_PATH = orig_meta
        Config.CACHE_TRAIN_FEATURES = orig_cache

    return train_dataset, val_dataset


def get_test_dataset():
    """
    Generates and loads Test dataset.

    Returns:
        tuple: (test_dataset, contact_ids)
    """
    # prepare_features uses Config.TEST_META_PATH when train_mode=False
    X_test, _, ids = prepare_features(train_mode=False)
    test_feats = _get_feature_names(Config.CACHE_TEST_FEATURES)

    test_dataset = ContactDataset(X_test, None, test_feats)

    return test_dataset, ids
