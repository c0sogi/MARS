import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def compute_auc(y_true, y_pred):
    """
    Safely computes ROC-AUC. Returns np.nan if y_true has only one class.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def power_mean(series, p):
    """
    Computes the generalized mean (power mean) of a series of numbers.
    M_p(x) = ( (1/N) * sum(x^p) )^(1/p)
    """
    # Filter out NaNs
    clean_series = np.array([x for x in series if not np.isnan(x)])
    if len(clean_series) == 0:
        return np.nan

    total = np.sum(np.power(clean_series, p))
    return np.power(total / len(clean_series), 1 / p)


def calculate_score(df: pd.DataFrame, pred_col: str):
    """
    Calculates the Jigsaw competition metric.

    Args:
        df: DataFrame containing targets, identity columns, and predictions.
        pred_col: Name of the column containing model predictions (probabilities).

    Returns:
        final_score: The weighted competition score.
        metrics_dict: A dictionary containing detailed metrics.
    """
    # Constants from Config
    target_col = Config.TARGET_COL
    identity_cols = Config.IDENTITY_COLS

    # 1. Convert to Boolean based on threshold 0.5
    # Note: The task description states target >= 0.5 is positive.
    # Standard practice for Jigsaw is identity >= 0.5 is considered a mention.
    y_true_bool = (df[target_col] >= 0.5).astype(int)
    y_pred = df[pred_col].values

    # Pre-calculate boolean identity columns for speed
    # We create a dictionary of boolean masks for each identity
    identity_masks = {
        ident: (df[ident] >= 0.5).astype(bool).values
        for ident in identity_cols
        if ident in df.columns
    }

    # 2. Overall AUC
    overall_auc = compute_auc(y_true_bool, y_pred)

    # 3. Calculate Bias AUCs per identity
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    # Dictionary to store per-identity raw metrics for debugging/logging
    per_identity_metrics = {}

    for ident in identity_cols:
        if ident not in identity_masks:
            continue

        mask_ident = identity_masks[ident]
        mask_toxic = y_true_bool == 1
        mask_nontoxic = y_true_bool == 0

        # --- Subgroup AUC ---
        # Restrict to examples mentioning the identity
        subgroup_mask = mask_ident
        s_auc = compute_auc(y_true_bool[subgroup_mask], y_pred[subgroup_mask])
        subgroup_aucs.append(s_auc)

        # --- BPSN AUC (Background Positive, Subgroup Negative) ---
        # Non-toxic examples mentioning identity (Subgroup Negative)
        # AND Toxic examples NOT mentioning identity (Background Positive)
        bpsn_mask = (mask_nontoxic & mask_ident) | (mask_toxic & ~mask_ident)
        bpsn_auc = compute_auc(y_true_bool[bpsn_mask], y_pred[bpsn_mask])
        bpsn_aucs.append(bpsn_auc)

        # --- BNSP AUC (Background Negative, Subgroup Positive) ---
        # Toxic examples mentioning identity (Subgroup Positive)
        # AND Non-toxic examples NOT mentioning identity (Background Negative)
        bnsp_mask = (mask_toxic & mask_ident) | (mask_nontoxic & ~mask_ident)
        bnsp_auc = compute_auc(y_true_bool[bnsp_mask], y_pred[bnsp_mask])
        bnsp_aucs.append(bnsp_auc)

        per_identity_metrics[ident] = {
            "subgroup_auc": s_auc,
            "bpsn_auc": bpsn_auc,
            "bnsp_auc": bnsp_auc,
        }

    # 4. Calculate Generalized Means (p = -5)
    p_value = -5
    gmean_subgroup = power_mean(subgroup_aucs, p_value)
    gmean_bpsn = power_mean(bpsn_aucs, p_value)
    gmean_bnsp = power_mean(bnsp_aucs, p_value)

    # 5. Final Score Calculation
    # Formula: w0*Overall + w1*Mean(Subgroup) + w2*Mean(BPSN) + w3*Mean(BNSP)
    # All weights are 0.25
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * gmean_subgroup)
        + (0.25 * gmean_bpsn)
        + (0.25 * gmean_bnsp)
    )

    # Compile results
    metrics_dict = {
        "final_score": final_score,
        "overall_auc": overall_auc,
        "subgroup_auc_mean": gmean_subgroup,
        "bpsn_auc_mean": gmean_bpsn,
        "bnsp_auc_mean": gmean_bnsp,
        "per_identity_breakdown": per_identity_metrics,
    }

    return final_score, metrics_dict
