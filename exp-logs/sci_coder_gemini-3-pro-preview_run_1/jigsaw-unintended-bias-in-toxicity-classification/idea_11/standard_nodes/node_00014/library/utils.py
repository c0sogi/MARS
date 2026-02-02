import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class JigsawMetric:
    """
    Implements the Jigsaw Unintended Bias in Toxicity Classification metric.
    Calculates Overall AUC and three bias-specific AUCs (Subgroup, BPSN, BNSP),
    aggregated via a generalized mean.
    """

    def __init__(self):
        self.identity_columns = Config.identity_cols
        # Competition specific parameters
        self.power = -5
        self.overall_weight = 0.25
        self.bias_weight = 0.25

    def compute_auc(self, y_true, y_pred):
        """
        Safe computation of ROC-AUC. Returns 0.5 if only one class is present.
        """
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5

    def power_mean(self, series, p):
        """
        Calculates the generalized mean (power mean) of a series.
        """
        total = np.sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def compute(self, valid_df, y_pred):
        """
        Computes the competition metric.

        Args:
            valid_df (pd.DataFrame): Validation dataframe containing 'binary_target'
                                     and identity columns.
            y_pred (np.array): Predicted probabilities for the positive class.

        Returns:
            dict: Dictionary containing the final score and sub-metrics.
        """
        # Create a working copy to avoid SettingWithCopy warnings
        eval_df = valid_df.copy()

        # Ensure predictions are aligned
        eval_df["prediction"] = y_pred

        # Use the binary target defined in Config
        target_col = Config.binary_target_col
        eval_df["label"] = eval_df[target_col].astype(int)

        # Convert fractional identity values to boolean for evaluation (Threshold >= 0.5)
        for col in self.identity_columns:
            eval_df[col] = (eval_df[col] >= 0.5).astype(bool)

        # 1. Overall AUC
        overall_auc = self.compute_auc(eval_df["label"], eval_df["prediction"])

        # 2. Compute Bias AUCs per subgroup
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        for identity in self.identity_columns:
            # A. Subgroup AUC
            # Restrict to examples mentioning the identity
            sub_mask = eval_df[identity] == True
            sub_df = eval_df[sub_mask]
            subgroup_auc = self.compute_auc(sub_df["label"], sub_df["prediction"])
            subgroup_aucs.append(subgroup_auc)

            # B. BPSN AUC (Background Positive, Subgroup Negative)
            # Restrict to (Non-Toxic & Identity) U (Toxic & No-Identity)
            # This measures if the model confuses non-toxic identity mentions with toxic comments.
            bpsn_mask = ((eval_df["label"] == 0) & (eval_df[identity] == True)) | (
                (eval_df["label"] == 1) & (eval_df[identity] == False)
            )
            bpsn_df = eval_df[bpsn_mask]
            bpsn_auc = self.compute_auc(bpsn_df["label"], bpsn_df["prediction"])
            bpsn_aucs.append(bpsn_auc)

            # C. BNSP AUC (Background Negative, Subgroup Positive)
            # Restrict to (Toxic & Identity) U (Non-Toxic & No-Identity)
            # This measures if the model confuses toxic identity mentions with non-toxic comments.
            bnsp_mask = ((eval_df["label"] == 1) & (eval_df[identity] == True)) | (
                (eval_df["label"] == 0) & (eval_df[identity] == False)
            )
            bnsp_df = eval_df[bnsp_mask]
            bnsp_auc = self.compute_auc(bnsp_df["label"], bnsp_df["prediction"])
            bnsp_aucs.append(bnsp_auc)

        # 3. Aggregate Bias AUCs using Generalized Mean (p = -5)
        # We assume the lists are not empty (guaranteed by Config.identity_cols)
        mp_subgroup = self.power_mean(subgroup_aucs, self.power)
        mp_bpsn = self.power_mean(bpsn_aucs, self.power)
        mp_bnsp = self.power_mean(bnsp_aucs, self.power)

        # 4. Calculate Final Score
        final_score = (
            self.overall_weight * overall_auc
            + self.bias_weight * mp_subgroup
            + self.bias_weight * mp_bpsn
            + self.bias_weight * mp_bnsp
        )

        return {
            "final_score": final_score,
            "overall_auc": overall_auc,
            "subgroup_auc_mean": mp_subgroup,
            "bpsn_auc_mean": mp_bpsn,
            "bnsp_auc_mean": mp_bnsp,
        }
