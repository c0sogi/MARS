import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from library.config import Config
from library.feature_engineering import FeatureEngineer


class DataManager:
    def __init__(self):
        self.config = Config
        self.fe = FeatureEngineer()

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def _apply_label_smoothing(self, df):
        """
        Applies Temporal Label Smoothing to the contact labels.
        Groups by player pair and applies a Gaussian filter to the binary contact sequence.

        Args:
            df (pd.DataFrame): Dataframe containing 'game_play', 'nfl_player_id_1',
                               'nfl_player_id_2', 'step', and 'contact'.

        Returns:
            pd.DataFrame: Dataframe with a new column 'contact_smooth'.
        """
        # We need to sort to ensure temporal order
        df = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Define a function to apply smoothing to a group
        sigma = self.config.LABEL_SMOOTHING_SIGMA

        def smooth_group(group):
            # Only smooth if we have enough data points
            if len(group) > 1:
                # Apply Gaussian filter
                # We use mode='nearest' to handle boundaries
                smoothed = gaussian_filter1d(
                    group["contact"].astype(float), sigma=sigma, mode="nearest"
                )
                return smoothed
            else:
                return group["contact"].astype(float).values

        # Apply to each pair
        # Grouping by player pair
        # Note: nfl_player_id_2 can be 'G' (ground), but in the features it might be int or sentinel.
        # FeatureEngineer converts G to sentinel or keeps ID.
        # In the merged DF, nfl_player_id_2 is likely numeric or mixed.
        # We cast to string for grouping safety.

        # Optimization: Vectorized approach or using transform is faster than apply on groups for large DFs
        # However, given the discontinuous nature of gated data, we must respect the group boundaries.
        # To speed this up, we only apply it to the training set which is manageable.

        # Create a unique pair ID for grouping
        pair_id = (
            df["game_play"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        )

        # We assign the result back.
        # Using transform with the custom function
        df["contact_smooth"] = df.groupby(pair_id)["contact"].transform(
            lambda x: gaussian_filter1d(x.astype(float), sigma=sigma, mode="nearest")
        )

        return df

    def get_train_features(self, load_cached_data=True):
        """
        Retrieves training features using the FeatureEngineer.
        """
        df = self.fe.generate_features(
            metadata_path=self.config.TRAIN_METADATA_PATH,
            tracking_path=self.config.TRAIN_TRACKING_PATH,
            mode="train",
            load_cached_data=load_cached_data,
        )
        return df

    def get_val_features(self, load_cached_data=True):
        """
        Retrieves validation features using the FeatureEngineer.
        """
        df = self.fe.generate_features(
            metadata_path=self.config.VAL_METADATA_PATH,
            tracking_path=self.config.TRAIN_TRACKING_PATH,  # Val data is in train tracking file
            mode="val",
            load_cached_data=load_cached_data,
        )
        return df

    def get_test_features(self, load_cached_data=True):
        """
        Retrieves test features using the FeatureEngineer.
        """
        df = self.fe.generate_features(
            metadata_path=self.config.TEST_METADATA_PATH,
            tracking_path=self.config.TEST_TRACKING_PATH,
            mode="test",
            load_cached_data=load_cached_data,
        )
        return df

    def get_scout_dataset(self, df):
        """
        Prepares a balanced dataset for training the Scout models (Phase 1).

        Args:
            df (pd.DataFrame): The full gated training dataframe.

        Returns:
            pd.DataFrame: A balanced subset (undersampled negatives).
        """
        # Separate positives and negatives
        positives = df[df["contact"] == 1]
        negatives = df[df["contact"] == 0]

        # Sample negatives to match positives (1:1 ratio for initial scout training)
        # or a fixed ratio. The Idea says "Balanced dataset".
        n_pos = len(positives)
        if len(negatives) > n_pos:
            negatives_sampled = negatives.sample(n=n_pos, random_state=self.config.SEED)
        else:
            negatives_sampled = negatives

        df_balanced = pd.concat([positives, negatives_sampled], axis=0).sample(
            frac=1, random_state=self.config.SEED
        )

        # Apply smoothing for training
        df_balanced = self._apply_label_smoothing(df_balanced)

        return df_balanced

    def get_anchored_dataset(self, df, hard_negative_indices):
        """
        Constructs the Anchored Dataset for the Expert model (Phase 3).

        Components:
        1. All Positives
        2. Mined Hard Negatives (provided via indices)
        3. Random Easy Negatives (Anchors) - determined by ANCHOR_RATIO

        Args:
            df (pd.DataFrame): The full gated training dataframe.
            hard_negative_indices (np.array or list): Indices of rows in df identified as hard negatives.

        Returns:
            pd.DataFrame: The constructed dataset.
        """
        # Ensure indices are unique
        hard_neg_idx = set(hard_negative_indices)

        # 1. Positives
        pos_mask = df["contact"] == 1
        df_pos = df[pos_mask]

        # 2. Hard Negatives
        # We must ensure these are actually negatives (just in case) and exist in the df
        # The indices passed should align with the df index.
        # We assume df has a standard RangeIndex or the indices match.
        # To be safe, we filter by index.
        df_hard_neg = df.loc[list(hard_neg_idx)]
        df_hard_neg = df_hard_neg[df_hard_neg["contact"] == 0]  # Safety check

        n_hard = len(df_hard_neg)

        # 3. Anchors (Random Easy Negatives)
        # Candidates are negatives that are NOT in the hard negative set
        neg_mask = df["contact"] == 0
        # Exclude hard negatives
        candidate_mask = neg_mask & (~df.index.isin(hard_neg_idx))
        df_candidates = df[candidate_mask]

        n_anchors = int(n_hard * self.config.ANCHOR_RATIO)

        if len(df_candidates) > n_anchors:
            df_anchors = df_candidates.sample(
                n=n_anchors, random_state=self.config.SEED
            )
        else:
            df_anchors = df_candidates

        # Combine
        df_final = pd.concat([df_pos, df_hard_neg, df_anchors], axis=0)

        # Shuffle
        df_final = df_final.sample(frac=1, random_state=self.config.SEED).reset_index(
            drop=True
        )

        # Apply Label Smoothing
        # Note: Smoothing works best on contiguous sequences.
        # The anchored dataset is highly fragmented.
        # However, the "Idea" specifies applying smoothing to address label noise.
        # If we smooth AFTER sampling, we lose temporal context.
        # Therefore, we should smooth the FULL dataset first, then sample.
        # BUT, smoothing is expensive.
        # Let's check the order: "Temporal Label Smoothing: Apply Gaussian smoothing... to the binary contact labels."
        # If we do it on the fragment, it's meaningless.
        # So we must apply it to the components before concatenation?
        # No, we must apply it to the full dataframe or the contiguous segments.

        # Strategy Update: Apply smoothing to the specific rows by looking up their context in the full DF?
        # Or simply apply smoothing to the full DF once at the beginning of the pipeline.
        # Since get_anchored_dataset takes 'df' (the full gated train set),
        # we can apply smoothing to 'df' inside this function (or helper) before slicing.

        # To avoid re-calculating smoothing every time, we should probably do it in get_train_features
        # but that function returns raw features.
        # Let's apply it here to the relevant subsets or the whole passed df.
        # Applying to the whole df (approx 1-2M rows) is feasible.

        print(
            "Applying Temporal Label Smoothing to full training set before slicing..."
        )
        df_smoothed = self._apply_label_smoothing(df.copy())

        # Now slice from the smoothed dataframe
        df_pos_s = df_smoothed[pos_mask]
        df_hard_neg_s = df_smoothed.loc[df_hard_neg.index]
        df_anchors_s = df_smoothed.loc[df_anchors.index]

        df_final = pd.concat([df_pos_s, df_hard_neg_s, df_anchors_s], axis=0)
        df_final = df_final.sample(frac=1, random_state=self.config.SEED).reset_index(
            drop=True
        )

        return df_final
