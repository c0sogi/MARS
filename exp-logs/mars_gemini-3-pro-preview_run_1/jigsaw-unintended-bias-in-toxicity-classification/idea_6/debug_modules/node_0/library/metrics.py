import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


class JigsawMetrics:
    """
    Implements the evaluation metrics for the Toxicity Classification task,
    specifically focusing on the ROC-AUC and Bias AUC metrics defined in the competition.
    """

    def __init__(self):
        self.identity_columns = Config.IDENTITY_COLUMNS
        # Weights as defined in the task: 0.25 for Overall, and 0.25 for each of the 3 bias aggregates
        self.weight_overall = 0.25
        self.weight_bias = 0.25
        self.power_p = -5

    def _compute_auc(self, y_true, y_score):
        """
        Safely computes ROC-AUC. Returns np.nan if only one class is present.
        """
        try:
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_score)
        except ValueError:
            return np.nan

    def _calculate_generalized_mean(self, scores, p):
        """
        Calculates the generalized mean (power mean) of a list of scores.
        Mp(ms) = (1/N * sum(ms^p))^(1/p)
        """
        scores = np.array([s for s in scores if not np.isnan(s)])
        if len(scores) == 0:
            return np.nan

        # Avoid division by zero or overflow issues with negative powers
        # We clip extremely small values to a small epsilon for stability if needed,
        # though AUCs are usually in [0.5, 1.0].
        scores = np.clip(scores, 1e-6, 1.0)

        mean_pow = np.mean(np.power(scores, p))
        return np.power(mean_pow, 1.0 / p)

    def calculate_bias_metrics(self, df, prediction_col, target_col, identity_col):
        """
        Calculates the three bias metrics for a single identity subgroup.

        Args:
            df: DataFrame containing boolean columns for target and identity.
            prediction_col: Name of the prediction column (probabilities).
            target_col: Name of the boolean target column.
            identity_col: Name of the boolean identity column.

        Returns:
            dict: { 'subgroup_auc': float, 'bpsn_auc': float, 'bnsp_auc': float }
        """
        # 1. Subgroup AUC
        # Restrict to examples that mention the identity
        subgroup_mask = df[identity_col]
        subgroup_auc = self._compute_auc(
            df[target_col][subgroup_mask], df[prediction_col][subgroup_mask]
        )

        # 2. BPSN (Background Positive, Subgroup Negative) AUC
        # Background Positive: Non-toxic examples that mention the identity
        # Subgroup Negative: Toxic examples that do not mention the identity
        # Note: The prompt nomenclature is slightly confusing.
        # BPSN usually stands for Background Positive, Subgroup Negative in terms of *Dataset Splits*,
        # but the prompt definition is:
        # "restrict ... to non-toxic examples that mention the identity AND toxic examples that do not."
        # This creates a binary classification task where:
        # Class 0: Non-toxic + Identity (Should be predicted low, but model might bias high)
        # Class 1: Toxic + No Identity (Should be predicted high)
        # If AUC is low, it means model confuses Non-toxic Identity as Toxic.

        bpsn_mask = (~df[target_col] & df[identity_col]) | (  # Non-toxic + Identity
            df[target_col] & ~df[identity_col]
        )  # Toxic + No Identity
        bpsn_auc = self._compute_auc(
            df[target_col][bpsn_mask], df[prediction_col][bpsn_mask]
        )

        # 3. BNSP (Background Negative, Subgroup Positive) AUC
        # "restrict ... to toxic examples that mention the identity AND non-toxic examples that do not."
        # Class 1: Toxic + Identity
        # Class 0: Non-toxic + No Identity

        bnsp_mask = (df[target_col] & df[identity_col]) | (  # Toxic + Identity
            ~df[target_col] & ~df[identity_col]
        )  # Non-toxic + No Identity
        bnsp_auc = self._compute_auc(
            df[target_col][bnsp_mask], df[prediction_col][bnsp_mask]
        )

        return {
            "subgroup_auc": subgroup_auc,
            "bpsn_auc": bpsn_auc,
            "bnsp_auc": bnsp_auc,
        }

    def calculate_score(self, val_df, prediction_col="prediction"):
        """
        Calculates the final competition score based on the validation dataframe.

        Args:
            val_df: Pandas DataFrame containing targets, identities, and predictions.
            prediction_col: Name of the column containing model probabilities.

        Returns:
            float: The final weighted score.
            dict: A dictionary containing the breakdown of metrics.
        """
        # Ensure we work on a copy to avoid modifying original df
        df = val_df.copy()

        # Convert continuous target to boolean (>= 0.5)
        # The prompt states: "test set examples with target >= 0.5 will be considered to be in the positive class"
        bool_target_col = "boolean_target"
        df[bool_target_col] = df[Config.TARGET_COL] >= 0.5

        # Convert identity columns to boolean (>= 0.5)
        # Standard practice for this dataset evaluation
        bool_identity_cols = []
        for col in self.identity_columns:
            bool_col = f"boolean_{col}"
            # Fill NaNs with 0 before thresholding
            df[bool_col] = df[col].fillna(0.0) >= 0.5
            bool_identity_cols.append(bool_col)

        # 1. Overall AUC
        overall_auc = self._compute_auc(df[bool_target_col], df[prediction_col])

        # 2. Calculate Bias AUCs per subgroup
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        per_identity_metrics = {}

        for i, ident_col in enumerate(self.identity_columns):
            bool_ident_col = bool_identity_cols[i]

            metrics = self.calculate_bias_metrics(
                df, prediction_col, bool_target_col, bool_ident_col
            )

            per_identity_metrics[ident_col] = metrics

            subgroup_aucs.append(metrics["subgroup_auc"])
            bpsn_aucs.append(metrics["bpsn_auc"])
            bnsp_aucs.append(metrics["bnsp_auc"])

        # 3. Calculate Generalized Means
        gen_mean_subgroup = self._calculate_generalized_mean(
            subgroup_aucs, self.power_p
        )
        gen_mean_bpsn = self._calculate_generalized_mean(bpsn_aucs, self.power_p)
        gen_mean_bnsp = self._calculate_generalized_mean(bnsp_aucs, self.power_p)

        # 4. Final Score
        # score = w0 * Overall + w1 * Mean(Subgroup) + w2 * Mean(BPSN) + w3 * Mean(BNSP)
        final_score = (
            self.weight_overall * overall_auc
            + self.weight_bias * gen_mean_subgroup
            + self.weight_bias * gen_mean_bpsn
            + self.weight_bias * gen_mean_bnsp
        )

        results = {
            "final_score": final_score,
            "overall_auc": overall_auc,
            "subgroup_auc_mean": gen_mean_subgroup,
            "bpsn_auc_mean": gen_mean_bpsn,
            "bnsp_auc_mean": gen_mean_bnsp,
            "per_identity_metrics": per_identity_metrics,
        }

        return final_score, results


def evaluate_predictions(val_df, prediction_col="prediction"):
    """
    Wrapper function to easily evaluate predictions from outside this module.
    """
    metrics = JigsawMetrics()
    score, detailed_results = metrics.calculate_score(val_df, prediction_col)

    print("=" * 40)
    print("EVALUATION METRICS")
    print("=" * 40)
    print(f"Overall AUC:       {detailed_results['overall_auc']:.16f}")
    print(f"Subgroup AUC Mean: {detailed_results['subgroup_auc_mean']:.16f}")
    print(f"BPSN AUC Mean:     {detailed_results['bpsn_auc_mean']:.16f}")
    print(f"BNSP AUC Mean:     {detailed_results['bnsp_auc_mean']:.16f}")
    print("-" * 40)
    print(f"FINAL SCORE:       {score:.16f}")
    print("=" * 40)

    return score, detailed_results
