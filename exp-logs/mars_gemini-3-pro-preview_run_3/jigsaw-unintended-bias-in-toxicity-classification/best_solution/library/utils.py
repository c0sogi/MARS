import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config, seed_everything


class JigsawEvaluator:
    """
    Evaluator class for the Jigsaw Unintended Bias in Toxicity Classification competition.
    Calculates the final weighted score combining Overall AUC and three bias-related AUC metrics
    (Subgroup, BPSN, BNSP) aggregated via a generalized mean.
    """

    def __init__(self, y_true, y_pred, identity_df):
        """
        Args:
            y_true (array-like): Ground truth targets (fractional or binary).
            y_pred (array-like): Predicted probabilities.
            identity_df (pd.DataFrame): DataFrame containing identity columns corresponding to y_true.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.df = identity_df.reset_index(drop=True)

        # Ensure inputs are aligned
        if len(self.y_true) != len(self.df):
            raise ValueError(
                f"Shape mismatch: y_true {len(self.y_true)} vs identity_df {len(self.df)}"
            )

        # Convert fractional targets to binary for ROC-AUC calculation
        # Competition standard: target >= 0.5 is positive
        self.y_binary = (self.y_true >= 0.5).astype(int)

    def _compute_auc(self, y_true, y_pred):
        """
        Computes ROC AUC score. Returns 0.5 if the subset has only one class.
        """
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5

    def _compute_generalized_mean(self, scores, p=-5):
        """
        Computes the generalized mean (power mean) of a list of scores.
        Mp(s) = (1/N * sum(s_i^p))^(1/p)
        """
        scores = np.array(scores)
        # Clip scores to avoid numerical issues (0^-5 is undefined)
        # AUC is typically > 0.5, but we clip to epsilon just in case
        scores = np.clip(scores, 1e-5, 1.0)
        mean_pow = np.mean(np.power(scores, p))
        return np.power(mean_pow, 1 / p)

    def get_final_metric(self):
        """
        Calculates the competition metric.

        Returns:
            final_score (float): The weighted score.
            metrics_dict (dict): Dictionary containing the breakdown of metrics.
        """
        # 1. Overall AUC
        overall_auc = self._compute_auc(self.y_binary, self.y_pred)

        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        # 2. Calculate Bias AUCs for each identity
        for identity in Config.IDENTITY_COLUMNS:
            if identity not in self.df.columns:
                continue

            # Identity mask: standard practice is value >= 0.5 implies mention
            # Fill NaNs with 0 (not mentioned)
            id_values = self.df[identity].fillna(0).values
            is_identity = id_values >= 0.5

            # --- Subgroup AUC ---
            # Restrict to examples mentioning the identity
            if np.sum(is_identity) > 0:
                s_auc = self._compute_auc(
                    self.y_binary[is_identity], self.y_pred[is_identity]
                )
            else:
                s_auc = 0.5
            subgroup_aucs.append(s_auc)

            # --- BPSN AUC (Background Positive, Subgroup Negative) ---
            # Non-toxic examples that mention identity AND Toxic examples that do not
            # (confuses non-toxic identity mentions with toxicity)
            is_toxic = self.y_binary == 1
            is_nontoxic = self.y_binary == 0

            mask_bpsn = (is_nontoxic & is_identity) | (is_toxic & ~is_identity)
            if np.sum(mask_bpsn) > 0:
                bpsn_auc = self._compute_auc(
                    self.y_binary[mask_bpsn], self.y_pred[mask_bpsn]
                )
            else:
                bpsn_auc = 0.5
            bpsn_aucs.append(bpsn_auc)

            # --- BNSP AUC (Background Negative, Subgroup Positive) ---
            # Toxic examples that mention identity AND Non-toxic examples that do not
            # (confuses toxic identity mentions with non-toxicity)
            mask_bnsp = (is_toxic & is_identity) | (is_nontoxic & ~is_identity)
            if np.sum(mask_bnsp) > 0:
                bnsp_auc = self._compute_auc(
                    self.y_binary[mask_bnsp], self.y_pred[mask_bnsp]
                )
            else:
                bnsp_auc = 0.5
            bnsp_aucs.append(bnsp_auc)

        # 3. Aggregate Bias Metrics using Generalized Mean (p=-5)
        if not subgroup_aucs:
            # Fallback if no identities are present (e.g., small debug subset)
            return overall_auc, {"overall_auc": overall_auc, "final_score": overall_auc}

        gen_mean_subgroup = self._compute_generalized_mean(subgroup_aucs)
        gen_mean_bpsn = self._compute_generalized_mean(bpsn_aucs)
        gen_mean_bnsp = self._compute_generalized_mean(bnsp_aucs)

        # 4. Final Weighted Score
        # Weights are 0.25 for each component
        final_score = (
            (0.25 * overall_auc)
            + (0.25 * gen_mean_subgroup)
            + (0.25 * gen_mean_bpsn)
            + (0.25 * gen_mean_bnsp)
        )

        return final_score, {
            "overall_auc": overall_auc,
            "subgroup_auc_mean": gen_mean_subgroup,
            "bpsn_auc_mean": gen_mean_bpsn,
            "bnsp_auc_mean": gen_mean_bnsp,
            "final_score": final_score,
        }
