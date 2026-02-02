import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import (
    IDENTITY_COLUMNS,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TEXT_PATH,
    TEST_TEXT_PATH,
)


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed_data(split, load_cached_data=True):
    """
    Loads the dataset for a specific split, merging metadata with text content.
    Implements caching to parquet to speed up subsequent loads.

    Args:
        split (str): One of 'train', 'validation', 'test'.
        load_cached_data (bool): If True, attempts to load from existing parquet cache.

    Returns:
        pd.DataFrame: The processed dataframe containing labels, identities, and text.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"{split}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. If not in cache, process from scratch
    # Determine file paths based on split
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
        source_text_path = TRAIN_TEXT_PATH
    elif split == "validation":
        meta_path = VAL_METADATA_PATH
        source_text_path = (
            TRAIN_TEXT_PATH  # Validation is a subset of original train file
        )
    elif split == "test":
        meta_path = TEST_METADATA_PATH
        source_text_path = TEST_TEXT_PATH
    else:
        raise ValueError(f"Invalid split name: {split}")

    # Load Metadata (Labels and IDs)
    meta_df = pd.read_csv(meta_path)

    # Load Text Source (IDs and Text)
    # Optimization: Read only necessary columns to save memory
    text_cols = ["id", "comment_text"]
    text_df = pd.read_csv(source_text_path, usecols=text_cols)

    # Merge Metadata with Text
    # meta_df is the left table (it defines the specific rows for this split)
    merged_df = meta_df.merge(text_df, on="id", how="left")

    # Fill any potential missing text values
    merged_df["comment_text"] = merged_df["comment_text"].fillna("")

    # 3. Save to cache
    merged_df.to_parquet(cache_path)

    return merged_df


class JigsawEvaluator:
    def __init__(self, y_true, y_pred, identity_df):
        """
        Evaluator class for calculating Jigsaw Unintended Bias metrics.

        Args:
            y_true (array-like): Ground truth toxicity targets (continuous or binary).
            y_pred (array-like): Predicted toxicity probabilities.
            identity_df (pd.DataFrame): DataFrame containing identity columns corresponding to y_true.
        """
        # Ensure inputs are numpy arrays
        if isinstance(y_true, (pd.Series, torch.Tensor)):
            y_true = (
                y_true.detach().cpu().numpy()
                if isinstance(y_true, torch.Tensor)
                else y_true.values
            )
        if isinstance(y_pred, (pd.Series, torch.Tensor)):
            y_pred = (
                y_pred.detach().cpu().numpy()
                if isinstance(y_pred, torch.Tensor)
                else y_pred.values
            )

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.identity_df = identity_df.reset_index(drop=True)

        # Binarize targets: Competition considers target >= 0.5 as toxic
        self.y_true_bin = (self.y_true >= 0.5).astype(int)

        # Binarize identities: Standard practice considers identity >= 0.5 as present
        self.identity_df_bin = (self.identity_df >= 0.5).astype(int)

    def _compute_auc(self, y_true, y_pred):
        """Helper to compute ROC-AUC safely."""
        try:
            # AUC is undefined if only one class is present
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, subgroup_col):
        """
        Calculates AUC on the subset of examples that mention the specific identity.
        """
        mask = self.identity_df_bin[subgroup_col] == 1
        if mask.sum() == 0:
            return np.nan
        return self._compute_auc(self.y_true_bin[mask], self.y_pred[mask])

    def _compute_bpsn_auc(self, subgroup_col):
        """
        Calculates Background Positive, Subgroup Negative (BPSN) AUC.
        Subset: (Non-toxic & Identity) U (Toxic & No-Identity)
        """
        subgroup = self.identity_df_bin[subgroup_col] == 1
        toxic = self.y_true_bin == 1

        # Non-toxic examples that mention identity (Label 0)
        mask_neg = (~toxic) & subgroup
        # Toxic examples that do not mention identity (Label 1)
        mask_pos = toxic & (~subgroup)

        mask = mask_neg | mask_pos
        if mask.sum() == 0:
            return np.nan

        return self._compute_auc(self.y_true_bin[mask], self.y_pred[mask])

    def _compute_bnsp_auc(self, subgroup_col):
        """
        Calculates Background Negative, Subgroup Positive (BNSP) AUC.
        Subset: (Toxic & Identity) U (Non-toxic & No-Identity)
        """
        subgroup = self.identity_df_bin[subgroup_col] == 1
        toxic = self.y_true_bin == 1

        # Toxic examples that mention identity (Label 1)
        mask_pos = toxic & subgroup
        # Non-toxic examples that do not mention identity (Label 0)
        mask_neg = (~toxic) & (~subgroup)

        mask = mask_pos | mask_neg
        if mask.sum() == 0:
            return np.nan

        return self._compute_auc(self.y_true_bin[mask], self.y_pred[mask])

    def _generalized_mean(self, scores, p=-5):
        """
        Calculates the generalized mean (power mean) of a list of scores.
        p = -5 penalizes low scores heavily.
        """
        valid_scores = [s for s in scores if not np.isnan(s)]
        if not valid_scores:
            return np.nan

        valid_scores = np.array(valid_scores)
        # Clip scores to avoid numerical instability with negative powers
        valid_scores = np.clip(valid_scores, 1e-6, 1.0)

        mean_pow = np.mean(np.power(valid_scores, p))
        return np.power(mean_pow, 1.0 / p)

    def get_final_metric(self):
        """
        Computes the final weighted score based on Overall AUC and Bias AUCs.

        Returns:
            tuple: (final_score, overall_auc, mean_subgroup_auc, mean_bpsn_auc, mean_bnsp_auc)
        """
        # 1. Overall AUC
        overall_auc = self._compute_auc(self.y_true_bin, self.y_pred)

        # 2. Per-Identity Bias AUCs
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        for col in IDENTITY_COLUMNS:
            if col in self.identity_df.columns:
                subgroup_aucs.append(self._compute_subgroup_auc(col))
                bpsn_aucs.append(self._compute_bpsn_auc(col))
                bnsp_aucs.append(self._compute_bnsp_auc(col))
            else:
                subgroup_aucs.append(np.nan)
                bpsn_aucs.append(np.nan)
                bnsp_aucs.append(np.nan)

        # 3. Generalized Means (p=-5)
        mean_subgroup = self._generalized_mean(subgroup_aucs)
        mean_bpsn = self._generalized_mean(bpsn_aucs)
        mean_bnsp = self._generalized_mean(bnsp_aucs)

        # 4. Final Weighted Score
        # Fill NaNs with 0.5 (random guess) for final summation if necessary,
        # though valid data should produce valid scores.
        m_sub = mean_subgroup if not np.isnan(mean_subgroup) else 0.5
        m_bpsn = mean_bpsn if not np.isnan(mean_bpsn) else 0.5
        m_bnsp = mean_bnsp if not np.isnan(mean_bnsp) else 0.5
        o_auc = overall_auc if not np.isnan(overall_auc) else 0.5

        final_score = 0.25 * o_auc + 0.25 * m_sub + 0.25 * m_bpsn + 0.25 * m_bnsp

        return final_score, overall_auc, mean_subgroup, mean_bpsn, mean_bnsp
