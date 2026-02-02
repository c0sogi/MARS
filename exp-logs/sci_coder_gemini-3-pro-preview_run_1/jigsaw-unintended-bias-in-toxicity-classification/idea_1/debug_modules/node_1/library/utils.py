import os
import random
import numpy as np
import pandas as pd
from sklearn import metrics


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class JigsawMetrics:
    """
    Implements the Jigsaw Unintended Bias in Toxicity Classification metrics.
    Calculates the overall AUC and three bias-specific AUCs (Subgroup, BPSN, BNSP)
    combined via a generalized mean.
    """

    def __init__(self):
        self.identities = [
            "male",
            "female",
            "homosexual_gay_or_lesbian",
            "christian",
            "jewish",
            "muslim",
            "black",
            "white",
            "psychiatric_or_mental_illness",
        ]

    def _compute_auc(self, y_true, y_pred):
        """
        Safely computes ROC-AUC. Returns 0.5 if only one class is present.
        """
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5

    def _power_mean(self, series, p):
        """
        Calculates the generalized mean (power mean) of a series.
        M_p(x) = ( (1/N) * sum(x^p) )^(1/p)
        """
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def compute_bias_metrics(self, df, y_pred, target_col="target"):
        """
        Computes the complete suite of bias metrics.

        Args:
            df (pd.DataFrame): Validation dataframe containing identity columns and target.
            y_pred (np.array): Predicted probabilities for the validation set.
            target_col (str): Name of the target column in df.

        Returns:
            dict: Dictionary containing the final score and individual sub-metrics.
        """
        # Ensure we work with a copy to avoid modifying the original df
        eval_df = df.copy()

        # Standardize columns to boolean for subset selection
        # Task description: "test set examples with target >= 0.5 will be considered positive"
        eval_df["bool_target"] = (eval_df[target_col] >= 0.5).astype(bool)
        eval_df["prediction"] = y_pred

        # Convert identity columns to boolean (standard practice is >= 0.5)
        for col in self.identities:
            if col in eval_df.columns:
                eval_df[col] = (eval_df[col] >= 0.5).astype(bool)
            else:
                # If column missing (shouldn't happen based on metadata), treat as False
                eval_df[col] = False

        # 1. Overall AUC
        overall_auc = self._compute_auc(eval_df["bool_target"], eval_df["prediction"])

        # 2. Per-Identity Bias AUCs
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        for identity in self.identities:
            # Subgroup AUC: Restrict to examples mentioning the identity
            subgroup_df = eval_df[eval_df[identity]]
            subgroup_auc = self._compute_auc(
                subgroup_df["bool_target"], subgroup_df["prediction"]
            )
            subgroup_aucs.append(subgroup_auc)

            # BPSN AUC: Background Positive, Subgroup Negative
            # Non-toxic examples that mention identity AND Toxic examples that do not
            # We want the model to distinguish these.
            # Positive Class in this subset: Toxic (no identity)
            # Negative Class in this subset: Non-Toxic (identity)
            bpsn_mask = (~eval_df["bool_target"] & eval_df[identity]) | (
                eval_df["bool_target"] & ~eval_df[identity]
            )
            bpsn_df = eval_df[bpsn_mask]
            bpsn_auc = self._compute_auc(bpsn_df["bool_target"], bpsn_df["prediction"])
            bpsn_aucs.append(bpsn_auc)

            # BNSP AUC: Background Negative, Subgroup Positive
            # Toxic examples that mention identity AND Non-toxic examples that do not
            # Positive Class in this subset: Toxic (identity)
            # Negative Class in this subset: Non-Toxic (no identity)
            bnsp_mask = (eval_df["bool_target"] & eval_df[identity]) | (
                ~eval_df["bool_target"] & ~eval_df[identity]
            )
            bnsp_df = eval_df[bnsp_mask]
            bnsp_auc = self._compute_auc(bnsp_df["bool_target"], bnsp_df["prediction"])
            bnsp_aucs.append(bnsp_auc)

        # 3. Generalized Means (p = -5)
        # We use a small epsilon or handle cases where AUC might be 0 to avoid division errors,
        # though AUC is usually > 0.5 for a working model.
        # If AUC is exactly 0, power mean with negative p explodes. Clip min AUC to 1e-6.
        subgroup_aucs = [max(x, 1e-6) for x in subgroup_aucs]
        bpsn_aucs = [max(x, 1e-6) for x in bpsn_aucs]
        bnsp_aucs = [max(x, 1e-6) for x in bnsp_aucs]

        mean_subgroup_auc = self._power_mean(subgroup_aucs, -5)
        mean_bpsn_auc = self._power_mean(bpsn_aucs, -5)
        mean_bnsp_auc = self._power_mean(bnsp_aucs, -5)

        # 4. Final Weighted Score
        final_score = (
            0.25 * overall_auc
            + 0.25 * mean_subgroup_auc
            + 0.25 * mean_bpsn_auc
            + 0.25 * mean_bnsp_auc
        )

        return {
            "score": final_score,
            "overall_auc": overall_auc,
            "subgroup_auc": mean_subgroup_auc,
            "bpsn_auc": mean_bpsn_auc,
            "bnsp_auc": mean_bnsp_auc,
            "per_subgroup_auc": dict(zip(self.identities, subgroup_aucs)),
            "per_bpsn_auc": dict(zip(self.identities, bpsn_aucs)),
            "per_bnsp_auc": dict(zip(self.identities, bnsp_aucs)),
        }
