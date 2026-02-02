import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def compute_generalized_mean(scores, p=-5):
    """
    Computes the generalized mean (power mean) of a list of scores.
    Formula: M_p(x) = ( (1/N) * sum(x^p) )^(1/p)

    Args:
        scores (list or np.array): List of AUC scores.
        p (float): Power parameter. Default is -5 per competition rules.

    Returns:
        float: The generalized mean. Returns np.nan if input is empty.
    """
    scores = np.array(scores)
    # Filter out NaNs to handle cases where a subgroup might be missing from the batch/set
    scores = scores[~np.isnan(scores)]

    if len(scores) == 0:
        return np.nan

    # Calculate power mean
    mean_pow = np.mean(np.power(scores, p))
    return np.power(mean_pow, 1.0 / p)


def calculate_jigsaw_metrics(
    val_df,
    prediction_col,
    target_col=Config.TARGET_COL,
    identity_columns=Config.IDENTITY_COLUMNS,
):
    """
    Calculates the Jigsaw Toxicity Classification competition metrics, including
    Overall AUC and the three Bias AUCs (Subgroup, BPSN, BNSP) aggregated via generalized mean.

    Args:
        val_df (pd.DataFrame): Validation dataframe containing targets, identities, and predictions.
        prediction_col (str): Column name for model predictions (probabilities).
        target_col (str): Column name for ground truth targets.
        identity_columns (list): List of identity column names to evaluate for bias.

    Returns:
        dict: A dictionary containing the 'final_score', 'overall_auc', and the three bias means.
    """
    # Work on a copy to avoid modifying the original dataframe
    df = val_df.copy()

    # 1. Prepare Binary Targets and Identities
    # Competition rule: target >= 0.5 is Positive (Toxic)
    df["binary_target"] = (df[target_col] >= 0.5).astype(int)

    # Pre-compute boolean masks for identities (>= 0.5 implies mention)
    for col in identity_columns:
        df[f"{col}_bool"] = df[col] >= 0.5

    # 2. Overall AUC
    try:
        overall_auc = roc_auc_score(df["binary_target"], df[prediction_col])
    except ValueError:
        overall_auc = np.nan

    # 3. Calculate Per-Identity Bias AUCs
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    for col in identity_columns:
        ident_col_bool = f"{col}_bool"

        # --- Subgroup AUC ---
        # Restrict to examples mentioning the identity
        subgroup_mask = df[ident_col_bool]

        # --- BPSN AUC (Background Positive, Subgroup Negative) ---
        # Background Positive: Toxic (1) AND No Identity
        # Subgroup Negative: Non-Toxic (0) AND Identity
        # We want the model to separate these two groups.
        bpsn_mask = ((df["binary_target"] == 1) & (~df[ident_col_bool])) | (
            (df["binary_target"] == 0) & (df[ident_col_bool])
        )

        # --- BNSP AUC (Background Negative, Subgroup Positive) ---
        # Background Negative: Non-Toxic (0) AND No Identity
        # Subgroup Positive: Toxic (1) AND Identity
        bnsp_mask = ((df["binary_target"] == 0) & (~df[ident_col_bool])) | (
            (df["binary_target"] == 1) & (df[ident_col_bool])
        )

        # Helper to safely calculate AUC on a subset
        def safe_auc(mask):
            subset = df[mask]
            # Need at least one positive and one negative example to calculate AUC
            if len(subset) == 0 or subset["binary_target"].nunique() < 2:
                return np.nan
            return roc_auc_score(subset["binary_target"], subset[prediction_col])

        subgroup_aucs.append(safe_auc(subgroup_mask))
        bpsn_aucs.append(safe_auc(bpsn_mask))
        bnsp_aucs.append(safe_auc(bnsp_mask))

    # 4. Compute Generalized Means (p = -5)
    p = -5
    mp_subgroup_auc = compute_generalized_mean(subgroup_aucs, p)
    mp_bpsn_auc = compute_generalized_mean(bpsn_aucs, p)
    mp_bnsp_auc = compute_generalized_mean(bnsp_aucs, p)

    # 5. Calculate Final Weighted Score
    # Score = 0.25*Overall + 0.25*Mp(Subgroup) + 0.25*Mp(BPSN) + 0.25*Mp(BNSP)
    # Handle NaNs in components by treating them as 0 if necessary, but usually data is sufficient.
    # If a mean is NaN, the score will be NaN, indicating an issue with evaluation data sufficiency.

    final_score = (
        (0.25 * overall_auc)
        + (0.25 * mp_subgroup_auc)
        + (0.25 * mp_bpsn_auc)
        + (0.25 * mp_bnsp_auc)
    )

    results = {
        "final_score": final_score,
        "overall_auc": overall_auc,
        "subgroup_auc_mean": mp_subgroup_auc,
        "bpsn_auc_mean": mp_bpsn_auc,
        "bnsp_auc_mean": mp_bnsp_auc,
    }

    return results
