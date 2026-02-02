import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class JigsawMetrics:
    """
    Implements the competition-specific bias evaluation metric.

    The final score is a weighted average of:
    1. Overall ROC-AUC
    2. Generalized Mean (p=-5) of Subgroup AUCs
    3. Generalized Mean (p=-5) of BPSN AUCs
    4. Generalized Mean (p=-5) of BNSP AUCs
    """

    def __init__(self):
        self.identity_columns = Config.IDENTITY_COLS
        self.target_col = Config.TARGET_COL
        self.p_value = -5
        self.weights = 0.25  # Equal weighting for all 4 components

    def _compute_auc(self, y_true, y_pred):
        """
        Calculates ROC-AUC safely. Returns NaN if only one class is present in y_true.
        """
        # Check if we have both positive and negative classes
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)

    def _power_mean(self, series, p):
        """
        Calculates the generalized mean (power mean) of a series.
        Ignores NaNs.

        Formula: (1/N * sum(x^p))^(1/p)
        """
        # Drop NaNs (subgroups that weren't present or had only 1 class)
        series = series.dropna()
        if len(series) == 0:
            return 0.0

        total = np.sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def compute(self, val_df, y_pred):
        """
        Computes the final Jigsaw score and sub-metrics.

        Args:
            val_df (pd.DataFrame): Validation dataframe containing target and identity columns.
            y_pred (np.array or list): Predicted probabilities for the positive class (toxic).

        Returns:
            dict: Dictionary containing 'score', 'overall_auc', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc'.
        """
        # Ensure y_pred is a flat numpy array
        y_pred = np.array(y_pred).flatten()

        # Binarize Target: >= 0.5 is Toxic (1), else Non-Toxic (0)
        y_true = (val_df[self.target_col].values >= 0.5).astype(int)

        # Binarize Identities: >= 0.5 is Present (1), else Absent (0)
        # We assume val_df contains the identity columns as floats [0, 1]
        identities = (val_df[self.identity_columns] >= 0.5).astype(int)

        # 1. Overall AUC
        overall_auc = self._compute_auc(y_true, y_pred)

        # Initialize lists to store per-identity AUCs
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        for col in self.identity_columns:
            # Create Boolean Masks
            id_mask = identities[col] == 1
            toxic_mask = y_true == 1
            nontoxic_mask = y_true == 0

            # --- Subgroup AUC ---
            # Restrict to examples mentioning the specific identity
            mask_sub = id_mask
            if mask_sub.sum() > 0:
                score = self._compute_auc(y_true[mask_sub], y_pred[mask_sub])
                subgroup_aucs.append(score)
            else:
                subgroup_aucs.append(np.nan)

            # --- BPSN AUC (Background Positive, Subgroup Negative) ---
            # Restrict to:
            # 1. Non-toxic examples mentioning identity (Subgroup Negative)
            # 2. Toxic examples NOT mentioning identity (Background Positive)
            # A low value means model confuses non-toxic identity mentions with toxic comments.
            sub_neg = id_mask & nontoxic_mask
            back_pos = (~id_mask) & toxic_mask
            mask_bpsn = sub_neg | back_pos

            if mask_bpsn.sum() > 0:
                score = self._compute_auc(y_true[mask_bpsn], y_pred[mask_bpsn])
                bpsn_aucs.append(score)
            else:
                bpsn_aucs.append(np.nan)

            # --- BNSP AUC (Background Negative, Subgroup Positive) ---
            # Restrict to:
            # 1. Toxic examples mentioning identity (Subgroup Positive)
            # 2. Non-toxic examples NOT mentioning identity (Background Negative)
            # A low value means model confuses toxic identity mentions with non-toxic comments.
            sub_pos = id_mask & toxic_mask
            back_neg = (~id_mask) & nontoxic_mask
            mask_bnsp = sub_pos | back_neg

            if mask_bnsp.sum() > 0:
                score = self._compute_auc(y_true[mask_bnsp], y_pred[mask_bnsp])
                bnsp_aucs.append(score)
            else:
                bnsp_aucs.append(np.nan)

        # Convert lists to Series for easier handling
        subgroup_aucs = pd.Series(subgroup_aucs)
        bpsn_aucs = pd.Series(bpsn_aucs)
        bnsp_aucs = pd.Series(bnsp_aucs)

        # Compute Generalized Means (p = -5)
        # This penalizes low performance on specific subgroups heavily
        subgroup_mean = self._power_mean(subgroup_aucs, self.p_value)
        bpsn_mean = self._power_mean(bpsn_aucs, self.p_value)
        bnsp_mean = self._power_mean(bnsp_aucs, self.p_value)

        # Final Weighted Score
        # score = 0.25*Overall + 0.25*Subgroup + 0.25*BPSN + 0.25*BNSP
        final_score = (
            (self.weights * overall_auc)
            + (self.weights * subgroup_mean)
            + (self.weights * bpsn_mean)
            + (self.weights * bnsp_mean)
        )

        return {
            "score": final_score,
            "overall_auc": overall_auc,
            "subgroup_auc": subgroup_mean,
            "bpsn_auc": bpsn_mean,
            "bnsp_auc": bnsp_mean,
        }
