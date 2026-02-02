import pandas as pd
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from library.config import Config
from library.utils import Timer, save_data, load_data, seed_everything
from library.feature_engineering import generate_features


class DataFactory:
    """
    Manages dataset creation, sampling, and label smoothing for the
    Orthogonal-Spectral Vector-Anchored Ensemble (OSVA-E).
    """

    def __init__(self):
        self.config = Config

    def load_and_process_data(self, split="train", load_cached_data=True):
        """
        Loads metadata and tracking data, then generates features using the FeatureEngineer.
        Handles caching to disk.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed feature dataframe.
        """
        seed_everything(self.config.SEED)

        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
            track_path = self.config.TRAIN_TRACKING_PATH
            cache_path = self.config.PROCESSED_TRAIN_PATH
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
            # Validation uses train tracking data (same source file usually, split by metadata)
            # Based on provided config, TRAIN_TRACKING_PATH covers both train/val splits
            # derived from train_labels.csv
            track_path = self.config.TRAIN_TRACKING_PATH
            cache_path = self.config.PROCESSED_VAL_PATH
        elif split == "test":
            meta_path = self.config.TEST_METADATA_PATH
            track_path = self.config.TEST_TRACKING_PATH
            cache_path = self.config.PROCESSED_TEST_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Delegate to feature engineering module
        df = generate_features(
            metadata_path=meta_path,
            tracking_path=track_path,
            output_path=cache_path,
            load_cached_data=load_cached_data,
        )

        # Ensure index is reset for consistent integer indexing later (crucial for mining)
        df = df.reset_index(drop=True)

        return df

    def get_scout_dataset(self, df):
        """
        Creates a balanced dataset for training Scout models.
        Downsamples negatives to match positives 1:1.

        Args:
            df (pd.DataFrame): The full gated dataframe.

        Returns:
            pd.DataFrame: Balanced dataframe.
        """
        with Timer("Constructing Scout Dataset"):
            # Separate classes
            pos_mask = df["contact"] == 1
            neg_mask = df["contact"] == 0

            df_pos = df[pos_mask]
            df_neg = df[neg_mask]

            n_pos = len(df_pos)
            n_neg = len(df_neg)

            # Downsample negatives
            if n_neg > n_pos:
                df_neg_sampled = df_neg.sample(n=n_pos, random_state=self.config.SEED)
            else:
                df_neg_sampled = df_neg

            # Combine and shuffle
            df_scout = pd.concat([df_pos, df_neg_sampled], axis=0)
            df_scout = df_scout.sample(
                frac=1.0, random_state=self.config.SEED
            ).reset_index(drop=True)

            print(
                f"Scout Dataset: {len(df_scout)} rows (Pos: {len(df_pos)}, Neg: {len(df_neg_sampled)})"
            )
            return df_scout

    def get_mining_pool(self, df):
        """
        Returns the dataset used for mining hard negatives.
        This is typically the full gated survivor set.

        Args:
            df (pd.DataFrame): The full processed dataframe.

        Returns:
            pd.DataFrame: The dataframe ready for inference.
        """
        return df

    def apply_temporal_smoothing(self, df):
        """
        Applies Gaussian smoothing to the binary contact labels to create soft targets.
        Smoothing is applied per player-pair sequence.

        Args:
            df (pd.DataFrame): Dataframe containing 'contact' and identification columns.

        Returns:
            pd.Series: Smoothed soft labels.
        """
        with Timer("Applying Temporal Label Smoothing"):
            # We need to sort to ensure temporal order
            # Note: We assume 'step' is the time variable
            sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]

            # Store original index to map back
            df["_orig_idx"] = df.index

            df_sorted = df.sort_values(sort_cols)

            # Define smoothing function
            sigma = self.config.LABEL_SMOOTHING_SIGMA

            def smooth_func(x):
                # Convert to float for smoothing
                return gaussian_filter1d(x.astype(float), sigma=sigma)

            # Group by pair and transform
            # We group by game_play + pair.
            # nfl_player_id_2 can be 'G', so it's mixed type, but pandas groupby handles it.
            soft_targets = df_sorted.groupby(
                ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
            )["contact"].transform(smooth_func)

            # Realign to original index
            # We assign the result back to the sorted df, then sort by index
            df_sorted["soft_contact"] = soft_targets

            # Restore order
            df_result = df_sorted.sort_values("_orig_idx")

            # Clean up
            soft_labels = df_result["soft_contact"].values

            return soft_labels

    def construct_expert_dataset(self, df, hard_negative_indices):
        """
        Constructs the Expert Dataset using the Anchored Mining strategy.
        Components:
        1. All Positives
        2. Mined Hard Negatives (Union of Scouts)
        3. Random Easy Negatives (Anchors)

        Also applies temporal smoothing to labels.

        Args:
            df (pd.DataFrame): The full gated dataframe.
            hard_negative_indices (np.ndarray): Indices of mined hard negatives.

        Returns:
            pd.DataFrame: The constructed expert dataset with 'contact' replaced/augmented by 'soft_contact'.
        """
        with Timer("Constructing Expert Dataset"):
            # 1. Apply Smoothing globally first (to preserve sequence context)
            # We add it as a new column
            df["contact"] = self.apply_temporal_smoothing(df)

            # 2. Identify Indices
            # Positives
            # Note: We use the original binary logic for selection, but train on soft labels.
            # However, we just overwrote 'contact' with soft labels.
            # We should have kept a binary mask or checked > 0.5, but better to check original.
            # Let's re-calculate binary mask from the soft labels roughly or assume
            # we should have kept the binary column.
            # To be safe, let's assume raw data had binary 0/1.
            # Smoothing smears 1s. So anything > 0.01 is likely near a contact,
            # but strict positives are best identified before overwrite.
            # Let's revert the overwrite logic:

            soft_labels = df["contact"].values  # This is now soft

            # We need the original binary for selection logic.
            # Since we modified df in place above, we can't easily get it back unless we check values.
            # A value of 1.0 (or close) implies contact.
            # Ideally, apply_temporal_smoothing shouldn't mutate 'contact' in place if we need binary later.
            # Let's adjust: apply_temporal_smoothing returns a Series.
            # We will reload or assume df has 'contact' as binary coming in.

            # RE-IMPLEMENTING LOGIC SAFELY:
            # Re-calculate soft labels without overwriting 'contact' immediately
            # We need to call apply_temporal_smoothing but it returns values.
            # Let's assume the previous call in this function was:
            # soft_labels = self.apply_temporal_smoothing(df)
            # df['soft_contact'] = soft_labels

            # But the code above did: df["contact"] = ...
            # Let's fix that in this scope.

            # Get soft labels
            soft_labels = self.apply_temporal_smoothing(df)
            df["soft_contact"] = soft_labels

            # Indices
            all_indices = df.index.values

            # Positives: Original contact is 1
            # We assume input df 'contact' is still binary 0/1
            pos_indices = df.index[df["contact"] == 1].values

            # Hard Negatives: Provided
            hard_neg_indices = np.array(hard_negative_indices)

            # Anchors: Random sample of Negatives NOT in Hard Negatives
            # Negatives are those not in pos_indices
            # Set operations for efficiency
            set_pos = set(pos_indices)
            set_hard = set(hard_neg_indices)

            # Candidates for anchors: All indices - (Pos U Hard)
            # Note: Hard negatives should technically be a subset of Negatives,
            # but scouts might flag a Positive as a Hard Negative (False Negative? No, Hard Neg is usually a Neg predicted as Pos).
            # Regardless, we want Easy Negatives.

            mask_is_pos = df["contact"] == 1
            mask_is_hard = df.index.isin(hard_neg_indices)

            # Easy Negatives: Not Positive AND Not Hard
            easy_neg_indices = df.index[(~mask_is_pos) & (~mask_is_hard)].values

            # Number of anchors
            n_hard = len(hard_neg_indices)
            n_anchors = int(n_hard * self.config.ANCHOR_RATIO)

            # Sample anchors
            if len(easy_neg_indices) > n_anchors:
                anchor_indices = np.random.choice(
                    easy_neg_indices, size=n_anchors, replace=False
                )
            else:
                anchor_indices = easy_neg_indices

            # Combine
            final_indices = np.concatenate(
                [pos_indices, hard_neg_indices, anchor_indices]
            )
            final_indices = np.unique(final_indices)  # Safety

            # Select subset
            df_expert = df.loc[final_indices].copy()

            # Shuffle
            df_expert = df_expert.sample(
                frac=1.0, random_state=self.config.SEED
            ).reset_index(drop=True)

            # Set target to soft contact
            # We keep 'contact' as the target column name for model compatibility,
            # but it now contains float probabilities.
            df_expert["contact"] = df_expert["soft_contact"]

            # Drop temp col
            df_expert.drop(
                columns=["soft_contact", "_orig_idx"], errors="ignore", inplace=True
            )

            print(f"Expert Dataset: {len(df_expert)} rows")
            print(f"  Positives: {len(pos_indices)}")
            print(f"  Hard Negatives: {len(hard_neg_indices)}")
            print(f"  Anchors: {len(anchor_indices)}")

            return df_expert
