import pandas as pd
import numpy as np
import scipy.ndimage
from library.config import Config
from library.features import FeatureGenerator
from library.utils import seed_everything


class NFLDataLoader:
    """
    Handles data loading, preprocessing, and dataset construction for the
    Dual-Basis Kinematic-Spectral Anchored-Ensemble (DB-KSAE).
    """

    def __init__(self):
        self.feature_gen = FeatureGenerator()
        seed_everything(Config.SEED)

    def load_features(self, stage="train", load_cached_data=True):
        """
        Loads the full feature set for the specified stage (train/val/test).
        Wrapper around FeatureGenerator.
        """
        if stage == "train":
            return self.feature_gen.process_train(load_cached_data=load_cached_data)
        elif stage == "val":
            return self.feature_gen.process_val(load_cached_data=load_cached_data)
        elif stage == "test":
            return self.feature_gen.process_test(load_cached_data=load_cached_data)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_scout_data(self, df):
        """
        Constructs the training data for the Scout models.
        Logic:
        1. Filter for Gated Survivors (gating_active == 1).
        2. Create a balanced dataset (1:1 Positives:Negatives).
        """
        # Filter by gating
        df_gated = df[df["gating_active"] == 1].copy()

        # Separate classes
        positives = df_gated[df_gated["contact"] == 1]
        negatives = df_gated[df_gated["contact"] == 0]

        # Downsample negatives to match positives
        n_pos = len(positives)
        if len(negatives) > n_pos:
            negatives = negatives.sample(n=n_pos, random_state=Config.SEED)

        # Combine and shuffle
        df_balanced = (
            pd.concat([positives, negatives])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]

        X = df_balanced[feature_cols]
        y = df_balanced["contact"]

        return X, y

    def get_mining_candidates(self, df):
        """
        Returns the features and indices of all Gated Survivors.
        Used by Scouts to mine Hard Negatives.
        """
        # Filter by gating
        df_gated = df[df["gating_active"] == 1].copy()

        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]

        # Return the subset dataframe so we can preserve indices
        return df_gated, df_gated[feature_cols]

    def apply_label_smoothing(self, df, sigma=1.0):
        """
        Applies Gaussian Temporal Label Smoothing to the 'contact' column.
        """
        # Ensure data is sorted by time within each pair
        # Sort keys: game_play, p1, p2, step
        df_sorted = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        ).copy()

        # Create a unique pair ID for faster grouping
        pair_ids = (
            df_sorted["game_play"].astype(str)
            + "_"
            + df_sorted["nfl_player_id_1"].astype(str)
            + "_"
            + df_sorted["nfl_player_id_2"].astype(str)
        )

        # Groupby and transform
        # We use the index to map back to the dataframe
        smoothed_values = df_sorted.groupby(pair_ids)["contact"].transform(
            lambda x: scipy.ndimage.gaussian_filter1d(
                x.astype(float), sigma=sigma, mode="nearest"
            )
        )

        df_sorted["contact"] = smoothed_values
        return df_sorted

    def get_expert_data(self, df, hard_negative_indices, anchor_ratio=1.0):
        """
        Constructs the training data for the Expert models.
        Components:
        1. All Positives (contact == 1)
        2. Mined Hard Negatives (from indices)
        3. Random Anchors (Easy Negatives, sampled from remaining gated survivors)

        Applies Temporal Label Smoothing to the targets.
        """
        # 1. Identify Positives
        positives = df[df["contact"] == 1].copy()

        # 2. Hard Negatives
        # hard_negative_indices are indices in 'df'
        hard_negatives = df.loc[df.index.isin(hard_negative_indices)].copy()

        # 3. Random Anchors
        # Candidates: Gated Active == 1, Contact == 0, NOT in Hard Negatives
        mask_candidates = (
            (df["gating_active"] == 1)
            & (df["contact"] == 0)
            & (~df.index.isin(hard_negative_indices))
        )

        anchor_candidates = df[mask_candidates]

        # Determine number of anchors
        n_anchors = int(len(positives) * anchor_ratio)

        if len(anchor_candidates) > n_anchors:
            anchors = anchor_candidates.sample(n=n_anchors, random_state=Config.SEED)
        else:
            anchors = anchor_candidates

        # Combine to get the target indices
        df_expert_subset = pd.concat([positives, hard_negatives, anchors], axis=0)

        # Remove duplicates just in case
        df_expert_subset = df_expert_subset[
            ~df_expert_subset.index.duplicated(keep="first")
        ]
        expert_indices = df_expert_subset.index

        # Apply Label Smoothing
        # CRITICAL: To smooth correctly, we must smooth the ENTIRE dataframe FIRST to preserve temporal context,
        # then select the rows.
        print("Applying Temporal Label Smoothing to full dataset before extraction...")
        df_smoothed = self.apply_label_smoothing(df, sigma=Config.LABEL_SMOOTHING_SIGMA)

        # Re-extract using indices from the smoothed dataframe
        df_final = df_smoothed.loc[expert_indices].copy()

        # Shuffle
        df_final = df_final.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]

        X = df_final[feature_cols]
        y = df_final["contact"]

        return X, y
