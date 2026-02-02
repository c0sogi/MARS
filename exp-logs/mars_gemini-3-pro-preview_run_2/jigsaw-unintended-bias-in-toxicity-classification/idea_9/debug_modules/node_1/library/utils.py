import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class JigsawEvaluator:
    def __init__(
        self,
        identity_columns,
        weight_overall=0.25,
        weight_subgroup=0.25,
        weight_bpsn=0.25,
        weight_bnsp=0.25,
        power=-5,
    ):
        self.identity_columns = identity_columns
        self.weights = [weight_overall, weight_subgroup, weight_bpsn, weight_bnsp]
        self.power = power

    def _compute_auc(self, y_true, y_pred):
        """
        Safely computes ROC-AUC. Returns 0.5 if only one class is present.
        """
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5

    def _generalized_mean(self, scores):
        """
        Calculates the generalized mean (power mean) of a list of scores.
        """
        scores = np.array(scores)
        # Clip scores to avoid zero for negative power (though AUC shouldn't be 0 usually)
        scores = np.clip(scores, 1e-6, 1.0)
        if self.power == 0:
            return np.exp(np.mean(np.log(scores)))
        return np.power(np.mean(np.power(scores, self.power)), 1.0 / self.power)

    def get_final_metric(self, y_true, y_pred, identities_df):
        """
        Calculates the competition metric.

        Args:
            y_true: Array-like of true targets (continuous or binary).
            y_pred: Array-like of predicted probabilities.
            identities_df: DataFrame containing identity columns.

        Returns:
            final_score: The weighted composite score.
            metrics: Dictionary containing individual metric components.
        """
        # Clip predictions to safe range as per task description to prevent divide-by-zero errors
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

        # Convert targets to binary for evaluation (Threshold >= 0.5)
        y_true_binary = (y_true >= 0.5).astype(int)

        # 1. Overall AUC
        overall_auc = self._compute_auc(y_true_binary, y_pred)

        # 2. Bias AUCs
        subgroup_scores = []
        bpsn_scores = []
        bnsp_scores = []

        # Ensure identities_df aligns with y_true/y_pred
        if isinstance(identities_df, pd.DataFrame):
            identities_df = identities_df.reset_index(drop=True)

        for col in self.identity_columns:
            if col not in identities_df.columns:
                continue

            # Identity mask (boolean)
            ident_mask = (identities_df[col] >= 0.5).values

            # Skip if no examples mention this identity
            if ident_mask.sum() == 0:
                continue

            # Subgroup AUC: Restrict to examples mentioning the identity
            sub_auc = self._compute_auc(y_true_binary[ident_mask], y_pred[ident_mask])
            subgroup_scores.append(sub_auc)

            # BPSN AUC (Background Positive, Subgroup Negative)
            # Restrict to: Non-toxic examples that mention identity (y=0, ident=1)
            #              AND Toxic examples that do not (y=1, ident=0)
            bpsn_mask = ((y_true_binary == 0) & ident_mask) | (
                (y_true_binary == 1) & (~ident_mask)
            )
            bpsn_auc = self._compute_auc(y_true_binary[bpsn_mask], y_pred[bpsn_mask])
            bpsn_scores.append(bpsn_auc)

            # BNSP AUC (Background Negative, Subgroup Positive)
            # Restrict to: Toxic examples that mention identity (y=1, ident=1)
            #              AND Non-toxic examples that do not (y=0, ident=0)
            bnsp_mask = ((y_true_binary == 1) & ident_mask) | (
                (y_true_binary == 0) & (~ident_mask)
            )
            bnsp_auc = self._compute_auc(y_true_binary[bnsp_mask], y_pred[bnsp_mask])
            bnsp_scores.append(bnsp_auc)

        # 3. Generalized Means
        if len(subgroup_scores) > 0:
            mp_subgroup = self._generalized_mean(subgroup_scores)
            mp_bpsn = self._generalized_mean(bpsn_scores)
            mp_bnsp = self._generalized_mean(bnsp_scores)
        else:
            mp_subgroup = 0.5
            mp_bpsn = 0.5
            mp_bnsp = 0.5

        # 4. Final Score
        final_score = (
            self.weights[0] * overall_auc
            + self.weights[1] * mp_subgroup
            + self.weights[2] * mp_bpsn
            + self.weights[3] * mp_bnsp
        )

        metrics = {
            "overall_auc": overall_auc,
            "subgroup_auc": mp_subgroup,
            "bpsn_auc": mp_bpsn,
            "bnsp_auc": mp_bnsp,
        }

        return final_score, metrics
