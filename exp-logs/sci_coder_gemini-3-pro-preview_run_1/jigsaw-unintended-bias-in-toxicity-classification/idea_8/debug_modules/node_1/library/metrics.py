import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


class JigsawEvaluator:
    """
    Evaluator class for the Jigsaw Unintended Bias in Toxicity Classification task.
    Calculates the Overall ROC-AUC and the specific Bias AUCs (Subgroup, BPSN, BNSP).
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing identity column names.
        """
        self.config = config
        self.identity_columns = config.aux_identity_cols
        self.target_col = config.target_col

    def _calculate_auc(self, y_true, y_pred):
        """
        Helper to calculate ROC-AUC safely.
        Returns 0.5 if only one class is present in y_true.
        """
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_pred)

    def _generalized_mean(self, scores, p=-5):
        """
        Calculates the generalized mean (power mean) of a list of scores.
        Formula: M_p(x) = ( (1/N) * sum(x_i^p) ) ^ (1/p)
        """
        scores = np.array(scores)
        # Clip to avoid numerical instability with negative powers if score is 0
        scores = np.clip(scores, 1e-6, 1.0)
        mean_pow = np.mean(np.power(scores, p))
        return np.power(mean_pow, 1.0 / p)

    def evaluate(self, valid_df: pd.DataFrame, preds: np.ndarray):
        """
        Calculates the competition metric.

        Args:
            valid_df (pd.DataFrame): Validation DataFrame containing targets and identities.
            preds (np.ndarray): Predicted probabilities for the positive class (toxicity).

        Returns:
            float: The final weighted score.
            dict: Dictionary containing the breakdown of metrics.
        """
        # Ensure binary target for evaluation (Threshold = 0.5)
        y_true = (valid_df[self.target_col] >= 0.5).astype(int).values
        y_pred = preds

        # 1. Overall AUC
        overall_auc = self._calculate_auc(y_true, y_pred)

        # 2. Bias AUCs per Identity
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        # Pre-calculate boolean masks for efficiency
        # Toxic mask
        is_toxic = y_true == 1

        # Identity masks (Identity is considered mentioned if value >= 0.5)
        # We fill NaNs with 0.0 to assume no mention if data is missing
        identity_matrix = valid_df[self.identity_columns].fillna(0.0).values
        is_identity_mentioned = identity_matrix >= 0.5

        for i, identity_col in enumerate(self.identity_columns):
            # Mask for current identity
            is_current_identity = is_identity_mentioned[:, i]

            # A. Subgroup AUC
            # Restriction: Only examples mentioning this identity
            mask_subgroup = is_current_identity
            if mask_subgroup.sum() > 0:
                s_auc = self._calculate_auc(
                    y_true[mask_subgroup], y_pred[mask_subgroup]
                )
            else:
                s_auc = 0.5
            subgroup_aucs.append(s_auc)

            # B. BPSN AUC (Background Positive, Subgroup Negative)
            # Background Positive: Toxic AND No Identity
            # Subgroup Negative: Non-Toxic AND Identity
            mask_bpsn = (is_toxic & ~is_current_identity) | (
                ~is_toxic & is_current_identity
            )
            if mask_bpsn.sum() > 0:
                bpsn_auc = self._calculate_auc(y_true[mask_bpsn], y_pred[mask_bpsn])
            else:
                bpsn_auc = 0.5
            bpsn_aucs.append(bpsn_auc)

            # C. BNSP AUC (Background Negative, Subgroup Positive)
            # Background Negative: Non-Toxic AND No Identity
            # Subgroup Positive: Toxic AND Identity
            mask_bnsp = (~is_toxic & ~is_current_identity) | (
                is_toxic & is_current_identity
            )
            if mask_bnsp.sum() > 0:
                bnsp_auc = self._calculate_auc(y_true[mask_bnsp], y_pred[mask_bnsp])
            else:
                bnsp_auc = 0.5
            bnsp_aucs.append(bnsp_auc)

        # 3. Aggregate Bias Metrics (Generalized Mean, p=-5)
        score_subgroup = self._generalized_mean(subgroup_aucs, p=-5)
        score_bpsn = self._generalized_mean(bpsn_aucs, p=-5)
        score_bnsp = self._generalized_mean(bnsp_aucs, p=-5)

        # 4. Final Weighted Score
        # score = 0.25*Overall + 0.25*Subgroup + 0.25*BPSN + 0.25*BNSP
        final_score = (
            0.25 * overall_auc
            + 0.25 * score_subgroup
            + 0.25 * score_bpsn
            + 0.25 * score_bnsp
        )

        metrics_dict = {
            "final_score": final_score,
            "overall_auc": overall_auc,
            "subgroup_auc_mean": score_subgroup,
            "bpsn_auc_mean": score_bpsn,
            "bnsp_auc_mean": score_bnsp,
        }

        return final_score, metrics_dict
