import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import (
    WORKING_DIR,
    KINEMATIC_COLS,
    CATEGORICAL_COLS,
    VISUAL_COLS,
    VISUAL_META_COLS,
    WINDOW_SIZE,
    BATCH_SIZE,
    TARGET_COL,
    SEED,
)
from library.data_processing import process_data
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the Entity-Augmented Residual-Visual Network.
    Serves Kinematic (Continuous + Categorical), Visual, and Gating features.
    """

    def __init__(self, X_kin, X_cat, X_vis, X_gate, y=None):
        self.X_kin = torch.FloatTensor(X_kin)
        self.X_cat = torch.LongTensor(X_cat)
        self.X_vis = torch.FloatTensor(X_vis)
        self.X_gate = torch.FloatTensor(X_gate)

        if y is not None:
            self.y = torch.FloatTensor(y).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        data = {
            "kinematic": self.X_kin[idx],
            "categorical": self.X_cat[idx],
            "visual": self.X_vis[idx],
            "gating": self.X_gate[idx],
        }
        if self.y is not None:
            return data, self.y[idx]
        return data


def engineer_features(df):
    """
    Performs in-memory feature engineering:
    1. Expands angular features (direction, orientation) into Sin/Cos components.
    2. Identifies and returns the lists of column names for each feature group.
    """

    # 1. Angular Continuity (Sin/Cos transform)
    # We need to find all direction/orientation columns across lags and players
    angle_base_cols = ["direction", "orientation"]

    # We will construct the final list of kinematic columns dynamically
    final_kin_cols = []

    # Iterate through the window and players to find columns and transform them
    for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"
        for player in ["_1", "_2"]:
            for col in KINEMATIC_COLS:
                col_name = f"{col}{suffix}{player}"

                if col in angle_base_cols:
                    # Create Sin/Cos features
                    sin_col = f"{col_name}_sin"
                    cos_col = f"{col_name}_cos"

                    # Convert degrees to radians and compute sin/cos
                    # Fill NaNs with 0 before transform (though process_data should have handled NaNs)
                    vals = np.deg2rad(df[col_name].fillna(0).values)
                    df[sin_col] = np.sin(vals)
                    df[cos_col] = np.cos(vals)

                    final_kin_cols.extend([sin_col, cos_col])
                else:
                    final_kin_cols.append(col_name)

    # 2. Categorical Columns
    final_cat_cols = []
    for player in ["_1", "_2"]:
        for col in CATEGORICAL_COLS:
            final_cat_cols.append(f"{col}{player}")

    # 3. Visual Columns (Current step only, as per process_data)
    final_vis_cols = []
    for player in ["_1", "_2"]:
        for col in VISUAL_COLS:
            final_vis_cols.append(f"{col}{player}")

    # 4. Gating Columns
    final_gate_cols = []
    for player in ["_1", "_2"]:
        # VISUAL_META_COLS usually contains 'view_available', 'box_area'
        # process_data creates 'view_available_1', 'box_area_1' etc.
        # We assume VISUAL_META_COLS in config are base names without suffix
        # But config says: "view_available", "box_area".
        # process_data explicitly creates them with suffixes.
        for col in VISUAL_META_COLS:
            final_gate_cols.append(f"{col}{player}")

    return df, final_kin_cols, final_cat_cols, final_vis_cols, final_gate_cols


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates data loading, processing, scaling, and loader creation.
    Implements caching for processed tensors.
    """

    modes = ["train", "validation", "test"]
    output_files = {}

    # Define cache file names
    for mode in modes:
        output_files[mode] = {
            "kin": os.path.join(WORKING_DIR, f"{mode}_X_kin.npy"),
            "cat": os.path.join(WORKING_DIR, f"{mode}_X_cat.npy"),
            "vis": os.path.join(WORKING_DIR, f"{mode}_X_vis.npy"),
            "gate": os.path.join(WORKING_DIR, f"{mode}_X_gate.npy"),
            "y": os.path.join(WORKING_DIR, f"{mode}_y.npy"),
        }

    # Check if all cache files exist
    all_cached = True
    if load_cached_data:
        for mode in modes:
            for key, path in output_files[mode].items():
                if not os.path.exists(path):
                    all_cached = False
                    break
    else:
        all_cached = False

    if all_cached:
        print("Loading processed tensors from cache...")
        data_tensors = {}
        for mode in modes:
            data_tensors[mode] = {k: np.load(v) for k, v in output_files[mode].items()}
    else:
        print("Processing data from scratch (generating tensors)...")

        # 1. Load DataFrames
        dfs = {}
        for mode in modes:
            dfs[mode] = process_data(mode=mode, load_cached_data=load_cached_data)

        # 2. Feature Engineering (Angular & Column Selection)
        # We need to determine column names from Train and apply to all
        print("Engineering features (Angular transforms)...")
        dfs["train"], kin_cols, cat_cols, vis_cols, gate_cols = engineer_features(
            dfs["train"]
        )
        dfs["validation"], _, _, _, _ = engineer_features(dfs["validation"])
        dfs["test"], _, _, _, _ = engineer_features(dfs["test"])

        # 3. Categorical Encoding
        print("Encoding categorical features...")
        # We need separate encoders for 'position' and 'team' usually,
        # but here we have flattened columns (pos_1, team_1, pos_2, team_2).
        # We should share encoders: one for 'position' columns, one for 'team' columns.

        # Helper to get all values for a category type to fit encoder
        # We fit on all available data (train+val+test) to handle unseen labels in splits
        pos_vals_list = []
        team_vals_list = []
        for mode in modes:
            pos_vals_list.extend([dfs[mode][f"position_{p}"] for p in [1, 2]])
            team_vals_list.extend([dfs[mode][f"team_{p}"] for p in [1, 2]])

        pos_vals = pd.concat(pos_vals_list).unique()
        team_vals = pd.concat(team_vals_list).unique()

        # Handle potential new values in val/test by ensuring 'Missing'/'Ground' are in vocab
        # process_data already fills NaNs with 'Missing' or 'Ground'

        pos_encoder = LabelEncoder().fit(pos_vals.astype(str))
        team_encoder = LabelEncoder().fit(team_vals.astype(str))

        # Transform
        for mode in modes:
            for p in [1, 2]:
                # Position
                col = f"position_{p}"
                # Use map/apply to handle unseen labels safely if strict
                # But here we assume closed set + 'Missing'.
                # For safety, we can map unseen to 'Missing' if we had a mapping dict.
                # Given strict LabelEncoder, we assume train covers all or we accept error.
                # NFL positions are standard.
                dfs[mode][col] = pos_encoder.transform(dfs[mode][col].astype(str))

                # Team
                col = f"team_{p}"
                dfs[mode][col] = team_encoder.transform(dfs[mode][col].astype(str))

        # 4. Scaling Continuous Features
        print("Scaling continuous features...")
        scaler_kin = StandardScaler()
        scaler_vis = StandardScaler()
        scaler_gate = StandardScaler()

        # Fit on Train
        scaler_kin.fit(dfs["train"][kin_cols])
        scaler_vis.fit(dfs["train"][vis_cols])
        scaler_gate.fit(dfs["train"][gate_cols])

        # Transform All and Extract Numpy
        data_tensors = {}
        for mode in modes:
            print(f"  Transforming {mode}...")
            X_kin = scaler_kin.transform(dfs[mode][kin_cols]).astype(np.float32)
            X_vis = scaler_vis.transform(dfs[mode][vis_cols]).astype(np.float32)
            X_gate = scaler_gate.transform(dfs[mode][gate_cols]).astype(np.float32)
            X_cat = dfs[mode][cat_cols].values.astype(np.int64)
            y = dfs[mode][TARGET_COL].values.astype(np.float32)

            data_tensors[mode] = {
                "kin": X_kin,
                "cat": X_cat,
                "vis": X_vis,
                "gate": X_gate,
                "y": y,
            }

            # Save to cache
            np.save(output_files[mode]["kin"], X_kin)
            np.save(output_files[mode]["cat"], X_cat)
            np.save(output_files[mode]["vis"], X_vis)
            np.save(output_files[mode]["gate"], X_gate)
            np.save(output_files[mode]["y"], y)

    # 5. Create Datasets and Loaders
    print("Creating DataLoaders...")

    train_dataset = ContactDataset(
        data_tensors["train"]["kin"],
        data_tensors["train"]["cat"],
        data_tensors["train"]["vis"],
        data_tensors["train"]["gate"],
        data_tensors["train"]["y"],
    )

    val_dataset = ContactDataset(
        data_tensors["validation"]["kin"],
        data_tensors["validation"]["cat"],
        data_tensors["validation"]["vis"],
        data_tensors["validation"]["gate"],
        data_tensors["validation"]["y"],
    )

    test_dataset = ContactDataset(
        data_tensors["test"]["kin"],
        data_tensors["test"]["cat"],
        data_tensors["test"]["vis"],
        data_tensors["test"]["gate"],
        data_tensors["test"]["y"],  # Test y is placeholder 0s
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Calculate input dimensions for model initialization
    input_dims = {
        "kinematic": data_tensors["train"]["kin"].shape[1],
        "categorical": len(CATEGORICAL_COLS) * 2,  # 2 players
        "visual": data_tensors["train"]["vis"].shape[1],
        "gating": data_tensors["train"]["gate"].shape[1],
        "vocab_sizes": {
            "position": int(
                data_tensors["train"]["cat"][:, [0, 2]].max() + 1
            ),  # approx max index
            "team": int(data_tensors["train"]["cat"][:, [1, 3]].max() + 1),
        },
    }

    # Refine vocab sizes exactly based on encoders if possible, but max index is safe for Embedding layer

    return train_loader, val_loader, test_loader, input_dims
