import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the SSE-RVN model.

    Attributes:
        X_kin (torch.Tensor): Continuous kinematic features (tracking lags, relative stats).
        X_vis (torch.Tensor): Continuous visual features (helmet box metrics).
        X_cat (torch.Tensor): Categorical features (team, position) for embeddings.
        y (torch.Tensor, optional): Binary target labels.
    """

    def __init__(self, X_kin, X_vis, X_cat, y=None):
        self.X_kin = torch.FloatTensor(X_kin)
        self.X_vis = torch.FloatTensor(X_vis)
        self.X_cat = torch.LongTensor(X_cat)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        sample = {
            "x_kin": self.X_kin[idx],
            "x_vis": self.X_vis[idx],
            "x_cat": self.X_cat[idx],
        }
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample


class DataProcessor:
    """
    Handles the preparation of data for the ContactDataset.
    Includes categorical recovery, encoding, and feature scaling.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        # Paths for artifacts
        self.scaler_kin_path = os.path.join(self.working_dir, "scaler_kin.joblib")
        self.scaler_vis_path = os.path.join(self.working_dir, "scaler_vis.joblib")
        self.encoder_team_path = os.path.join(self.working_dir, "encoder_team.joblib")
        self.encoder_pos_path = os.path.join(self.working_dir, "encoder_pos.joblib")

    def _get_tracking_path(self, split):
        # Map split to the correct raw tracking file
        fname = (
            "test_player_tracking.csv"
            if split == "test"
            else "train_player_tracking.csv"
        )
        return os.path.join(Config.INPUT_DIR, fname)

    def _get_categorical_map(self, split):
        """
        Generates or loads a cached mapping of (game_play, player_id) -> (team, position).
        Required because feature_engineering.py does not preserve these static columns.
        """
        cache_path = os.path.join(self.working_dir, f"cat_map_{split}.parquet")

        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        # Load raw tracking data to extract static attributes
        tracking_path = self._get_tracking_path(split)
        use_cols = ["game_play", "nfl_player_id", "team", "position"]

        # Read only necessary columns
        df_tr = pd.read_csv(tracking_path, usecols=use_cols)

        # Drop duplicates to get unique player info per play
        # (Players don't change team/position mid-play)
        df_map = df_tr.drop_duplicates(subset=["game_play", "nfl_player_id"]).copy()

        # Save to cache
        df_map.to_parquet(cache_path, index=False)
        return df_map

    def enrich_features(self, df, split):
        """
        Merges team and position information into the processed features dataframe.
        """
        df_map = self._get_categorical_map(split)

        # 1. Merge for Player 1
        # Rename map columns for P1
        map_p1 = df_map.rename(
            columns={
                "nfl_player_id": "nfl_player_id_1",
                "team": "team_1",
                "position": "position_1",
            }
        )

        # Ensure ID types match (df is likely float from merge, map is int)
        df["nfl_player_id_1"] = pd.to_numeric(df["nfl_player_id_1"], errors="coerce")

        df = df.merge(map_p1, on=["game_play", "nfl_player_id_1"], how="left")

        # 2. Merge for Player 2
        # Handle 'G' (Ground) in P2 ID
        df["nfl_player_id_2_temp"] = pd.to_numeric(
            df["nfl_player_id_2"], errors="coerce"
        )

        map_p2 = df_map.rename(
            columns={
                "nfl_player_id": "nfl_player_id_2_temp",
                "team": "team_2",
                "position": "position_2",
            }
        )

        df = df.merge(
            map_p2,
            left_on=["game_play", "nfl_player_id_2_temp"],
            right_on=["game_play", "nfl_player_id_2_temp"],
            how="left",
        )

        # 3. Fill Missing Values
        # P1 missing: likely tracking mismatch, fill UNK
        df["team_1"] = df["team_1"].fillna("UNK")
        df["position_1"] = df["position_1"].fillna("UNK")

        # P2 missing: includes Ground and mismatches
        is_ground = df["nfl_player_id_2"] == "G"

        # Explicitly label Ground
        df.loc[is_ground, "team_2"] = "Ground"
        df.loc[is_ground, "position_2"] = "Ground"

        # Fill remaining P2 missing with UNK
        df["team_2"] = df["team_2"].fillna("UNK")
        df["position_2"] = df["position_2"].fillna("UNK")

        # Cleanup
        df = df.drop(columns=["nfl_player_id_2_temp"])

        return df

    def get_dataset(self, df, split="train", fit_scalers=False):
        """
        Main pipeline: Enrich -> Encode -> Scale -> Create Dataset.

        Args:
            df (pd.DataFrame): Processed feature dataframe.
            split (str): 'train', 'test', etc.
            fit_scalers (bool): Whether to fit new scalers/encoders (True for training set).

        Returns:
            ContactDataset: Ready for DataLoader.
        """
        # 1. Enrich with categorical data
        df = self.enrich_features(df, split)

        # 2. Define Column Groups
        # Metadata/Keys to exclude from features
        exclude_cols = {
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
            "team_1",
            "position_1",
            "team_2",
            "position_2",
            "path_endzone",
            "path_sideline",
            "path_all29",
            "view_1",
            "view_2",
        }

        # Visual Columns (explicitly defined in feature engineering)
        vis_base = ["left", "top", "width", "height"]
        vis_cols = [f"{c}_1" for c in vis_base] + [f"{c}_2" for c in vis_base]

        # Kinematic Columns: Everything else
        kin_cols = [
            c for c in df.columns if c not in exclude_cols and c not in vis_cols
        ]

        # Sort to ensure deterministic order
        kin_cols = sorted(kin_cols)
        vis_cols = sorted(vis_cols)

        # 3. Handle Categorical Encoding
        cat_cols = ["team_1", "position_1", "team_2", "position_2"]

        if fit_scalers:
            le_team = LabelEncoder()
            le_pos = LabelEncoder()

            # Fit on union of P1 and P2 to cover all possible values
            all_teams = pd.concat([df["team_1"], df["team_2"]]).unique().astype(str)
            all_pos = (
                pd.concat([df["position_1"], df["position_2"]]).unique().astype(str)
            )

            le_team.fit(all_teams)
            le_pos.fit(all_pos)

            joblib.dump(le_team, self.encoder_team_path)
            joblib.dump(le_pos, self.encoder_pos_path)
        else:
            le_team = joblib.load(self.encoder_team_path)
            le_pos = joblib.load(self.encoder_pos_path)

        # Transform Categoricals with fallback for unseen labels
        X_cat_list = []
        encoders = [le_team, le_pos, le_team, le_pos]

        for col, le in zip(cat_cols, encoders):
            s = df[col].astype(str)
            known_classes = set(le.classes_)
            # Map unknown classes to the first class (usually arbitrary, but safe)
            # In a real scenario, we might want a dedicated 'UNK' token fitted during training
            fallback_class = le.classes_[0]
            s_mapped = s.apply(lambda x: x if x in known_classes else fallback_class)
            X_cat_list.append(le.transform(s_mapped))

        X_cat = np.stack(X_cat_list, axis=1)  # Shape: (N, 4)

        # 4. Handle Continuous Scaling
        X_kin = df[kin_cols].values.astype(np.float32)
        X_vis = df[vis_cols].values.astype(np.float32)

        if fit_scalers:
            scaler_kin = StandardScaler()
            scaler_vis = StandardScaler()

            scaler_kin.fit(X_kin)
            scaler_vis.fit(X_vis)

            joblib.dump(scaler_kin, self.scaler_kin_path)
            joblib.dump(scaler_vis, self.scaler_vis_path)
        else:
            scaler_kin = joblib.load(self.scaler_kin_path)
            scaler_vis = joblib.load(self.scaler_vis_path)

        X_kin = scaler_kin.transform(X_kin)
        X_vis = scaler_vis.transform(X_vis)

        # 5. Extract Targets
        y = df["contact"].values.astype(np.float32) if "contact" in df.columns else None

        return ContactDataset(X_kin, X_vis, X_cat, y)
