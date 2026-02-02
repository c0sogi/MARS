import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


class BiasMetricCalculator:
    """
    Calculates the Jigsaw Unintended Bias Toxicity Classification metrics.

    Metrics include:
    1. Overall AUC
    2. Subgroup AUC (per identity)
    3. BPSN AUC (Background Positive, Subgroup Negative)
    4. BNSP AUC (Background Negative, Subgroup Positive)
    5. Generalized Mean of Bias AUCs (p = -5)
    6. Final Weighted Score
    """

    def __init__(self, identity_columns=None):
        """
        Args:
            identity_columns (list): List of identity column names.
                                     Defaults to Config.IDENTITY_COLS.
        """
        self.identity_columns = (
            identity_columns if identity_columns is not None else Config.IDENTITY_COLS
        )

    def _compute_auc(self, y_true, y_pred):
        """
        Helper to compute ROC-AUC safely.
        Returns np.nan if only one class is present in y_true.
        """
        try:
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_generalized_mean(self, scores, p=-5):
        """
        Computes the generalized mean of a list of scores.
        Formula: M_p(x) = (1/N * sum(x^p))^(1/p)
        """
        scores = np.array(scores)
        # Filter out NaNs
        scores = scores[~np.isnan(scores)]

        if len(scores) == 0:
            return np.nan

        # Avoid division by zero or overflow issues with extreme values
        # AUC is typically in [0, 1], usually > 0.5.
        # We clip slightly above 0 to prevent inf with negative power.
        scores = np.clip(scores, 1e-6, 1.0)

        mean_pow = np.mean(np.power(scores, p))
        return np.power(mean_pow, 1 / p)

    def calculate_bias_metrics(self, y_true, y_pred, identities):
        """
        Computes all bias metrics and the final competition score.

        Args:
            y_true (np.ndarray): Ground truth targets (continuous or binary).
            y_pred (np.ndarray): Predicted probabilities.
            identities (np.ndarray or pd.DataFrame): Identity attributes.
                If np.ndarray, assumed to be in order of self.identity_columns.

        Returns:
            dict: Dictionary containing 'final_score', 'overall_auc', and detailed breakdown.
            pd.DataFrame: A dataframe with per-identity metrics.
        """
        # 1. Standardize Inputs
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Binarize targets (Threshold 0.5 as per competition rules)
        y_true_bin = (y_true >= 0.5).astype(int)

        # Handle identities input
        if isinstance(identities, pd.DataFrame):
            identity_df = identities[self.identity_columns].copy()
        else:
            # Assume numpy array matches column order
            identities = np.array(identities)
            if identities.shape[1] != len(self.identity_columns):
                raise ValueError(
                    f"Identity matrix shape {identities.shape} does not match "
                    f"number of identity columns {len(self.identity_columns)}."
                )
            identity_df = pd.DataFrame(identities, columns=self.identity_columns)

        # Binarize identities (Threshold 0.5)
        # We treat an identity as 'mentioned' if the annotator score is >= 0.5
        identity_bool = identity_df >= 0.5

        # 2. Compute Overall AUC
        overall_auc = self._compute_auc(y_true_bin, y_pred)

        # 3. Compute Per-Subgroup Metrics
        records = []

        for col in self.identity_columns:
            mask_subgroup = identity_bool[col].values

            # A. Subgroup AUC
            # Restrict to examples mentioning the identity
            sub_auc = self._compute_auc(
                y_true_bin[mask_subgroup], y_pred[mask_subgroup]
            )

            # B. BPSN AUC (Background Positive, Subgroup Negative)
            # Non-toxic examples that mention the identity AND Toxic examples that do not
            # Likely confuses non-toxic identity mentions with toxic comments
            mask_bpsn = ((y_true_bin == 0) & mask_subgroup) | (
                (y_true_bin == 1) & (~mask_subgroup)
            )
            bpsn_auc = self._compute_auc(y_true_bin[mask_bpsn], y_pred[mask_bpsn])

            # C. BNSP AUC (Background Negative, Subgroup Positive)
            # Toxic examples that mention the identity AND Non-toxic examples that do not
            # Likely confuses toxic identity mentions with non-toxic comments
            mask_bnsp = ((y_true_bin == 1) & mask_subgroup) | (
                (y_true_bin == 0) & (~mask_subgroup)
            )
            bnsp_auc = self._compute_auc(y_true_bin[mask_bnsp], y_pred[mask_bnsp])

            records.append(
                {
                    "subgroup": col,
                    "subgroup_auc": sub_auc,
                    "bpsn_auc": bpsn_auc,
                    "bnsp_auc": bnsp_auc,
                }
            )

        metrics_df = pd.DataFrame(records)

        # 4. Compute Generalized Means (p = -5)
        mp_subgroup = self._compute_generalized_mean(metrics_df["subgroup_auc"])
        mp_bpsn = self._compute_generalized_mean(metrics_df["bpsn_auc"])
        mp_bnsp = self._compute_generalized_mean(metrics_df["bnsp_auc"])

        # 5. Calculate Final Score
        # score = w0*Overall + w1*Mp(Sub) + w2*Mp(BPSN) + w3*Mp(BNSP)
        # All weights = 0.25
        final_score = (
            (0.25 * overall_auc)
            + (0.25 * mp_subgroup)
            + (0.25 * mp_bpsn)
            + (0.25 * mp_bnsp)
        )

        results = {
            "final_score": final_score,
            "overall_auc": overall_auc,
            "mp_subgroup_auc": mp_subgroup,
            "mp_bpsn_auc": mp_bpsn,
            "mp_bnsp_auc": mp_bnsp,
        }

        return results, metrics_df


def calculate_score(y_true, y_pred, identities):
    """
    Wrapper function to calculate the score using the default configuration.

    Args:
        y_true: Ground truth targets.
        y_pred: Model predictions.
        identities: Identity matrix.

    Returns:
        float: The final competition score.
    """
    calculator = BiasMetricCalculator()
    results, _ = calculator.calculate_bias_metrics(y_true, y_pred, identities)
    return results["final_score"]
