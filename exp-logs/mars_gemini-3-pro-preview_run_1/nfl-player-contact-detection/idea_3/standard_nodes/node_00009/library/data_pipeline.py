import pandas as pd
import numpy as np
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.utils import seed_everything


class DataPipeline:
    """
    Orchestrates data preparation for the Split-Stream Kinematic Architecture.
    Interfaces with FeatureEngineer to retrieve processed features and applies
    high-level dataset management (splitting, subsampling, formatting).
    """

    def __init__(self, config=Config):
        self.config = config
        self.fe = FeatureEngineer(config)
        seed_everything(self.config.SEED)

    def get_unified_data(self, split="train", load_cached=True, subsample_size=None):
        """
        Prepares data for the Unified Model.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to use cached feature files.
            subsample_size (int, optional): If set and split is 'train', downsamples.

        Returns:
            pd.DataFrame: The dataset containing metadata, features, and target.
            list: List of column names to be used as features for the model.
        """
        # 1. Retrieve Feature-Engineered Data
        df = self.fe.create_unified_features(split=split, load_cached=load_cached)

        # 2. Identify Feature Columns
        feature_cols = []
        for col in df.columns:
            is_feature = False
            for base_feat in self.config.UNIFIED_FEATURES:
                if col.startswith(base_feat):
                    is_feature = True
                    break
            if is_feature:
                feature_cols.append(col)

        # 3. Apply Subsampling (Train only)
        if split == "train" and subsample_size is not None and subsample_size < len(df):
            print(f"Subsampling Unified data to ~{subsample_size} rows...")
            pos_df = df[df["contact"] == 1]
            neg_df = df[df["contact"] == 0]

            n_pos = len(pos_df)
            n_neg_target = subsample_size - n_pos

            if n_neg_target > 0:
                neg_df_sampled = neg_df.sample(
                    n=min(len(neg_df), n_neg_target), random_state=self.config.SEED
                )
                df = pd.concat([pos_df, neg_df_sampled], axis=0)
                df = df.sample(frac=1, random_state=self.config.SEED).reset_index(
                    drop=True
                )
            else:
                df = pos_df.sample(n=subsample_size, random_state=self.config.SEED)

            print(f"Subsampled data shape: {df.shape}")
            print(f"Class balance: {df['contact'].mean():.4f}")

        return df, feature_cols

    def get_stream_a_data(self, split="train", subsample_size=None):
        """
        Prepares data for Stream A (Interaction Model).
        Filters Unified data for Player-Player interactions.
        """
        # Load full unified data (no subsampling yet)
        df, feature_cols = self.get_unified_data(
            split=split, load_cached=True, subsample_size=None
        )

        # Filter for Player-Player (is_ground == 0)
        df = df[df["is_ground"] == 0].copy()

        # Apply Subsampling (Train only)
        if split == "train" and subsample_size is not None and subsample_size < len(df):
            print(f"Subsampling Stream A data to ~{subsample_size} rows...")
            pos_df = df[df["contact"] == 1]
            neg_df = df[df["contact"] == 0]

            n_pos = len(pos_df)
            n_neg_target = subsample_size - n_pos

            if n_neg_target > 0:
                neg_df_sampled = neg_df.sample(
                    n=min(len(neg_df), n_neg_target), random_state=self.config.SEED
                )
                df = pd.concat([pos_df, neg_df_sampled], axis=0)
                df = df.sample(frac=1, random_state=self.config.SEED).reset_index(
                    drop=True
                )
            else:
                df = pos_df.sample(n=subsample_size, random_state=self.config.SEED)

            print(f"Subsampled Stream A shape: {df.shape}")

        return df, feature_cols

    def get_stream_b_data(self, split="train"):
        """
        Prepares data for Stream B (Impact Model).
        Filters for Player-Ground interactions and formats as (N, C, T) tensor.
        """
        # 1. Get Paths
        meta_path, track_path = self.fe._get_paths(split)

        # 2. Load Metadata and Filter for Ground
        df_meta = pd.read_csv(meta_path)
        # Ensure keys
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)

        # Filter: nfl_player_id_2 == 'G'
        df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        if df_meta.empty:
            print(f"Warning: No ground contacts found in {split} split.")
            return np.array([]), np.array([]), df_meta

        # 3. Get Wide Tracking Data
        # This contains lag features for all players
        df_track = self.fe._process_tracking_data(track_path)

        # 4. Merge
        print(f"Merging Stream B data for {len(df_meta)} interactions...")
        df = pd.merge(
            df_meta,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 5. Construct Tensor (N, Channels, Time)
        features = self.config.STREAM_B_FEATURES
        window = self.config.WINDOW_SIZE

        tensor_list = []

        for feat in features:
            # Construct temporal column list: minus_W ... 0 ... W
            cols = []
            # Past
            for k in range(window, 0, -1):
                cols.append(f"{feat}_minus_{k}")
            # Present
            cols.append(f"{feat}_0")
            # Future
            for k in range(1, window + 1):
                cols.append(f"{feat}_{k}")

            # Extract and fill NaNs
            # Shape: (N, T)
            data = df[cols].fillna(0).values.astype(np.float32)
            tensor_list.append(data)

        # Stack to (N, C, T)
        X = np.stack(tensor_list, axis=1)

        # Targets
        y = df["contact"].values.astype(np.float32) if "contact" in df.columns else None

        return X, y, df
