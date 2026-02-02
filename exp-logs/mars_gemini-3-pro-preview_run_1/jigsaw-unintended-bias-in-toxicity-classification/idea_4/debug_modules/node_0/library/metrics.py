import numpy as np
import pandas as pd
from sklearn import metrics
from library.config import Config


def calculate_auc(y_true, y_pred):
    """
    Calculates the ROC-AUC score safely.
    Returns 0.5 if the subset has only one class or is empty.
    """
    try:
        # Check if we have both classes represented
        if len(np.unique(y_true)) < 2:
            return 0.5
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def power_mean(series, p):
    """
    Calculates the generalized mean (power mean) of a series.
    M_p(x) = ( (1/N) * sum(x^p) ) ^ (1/p)
    """
    # Avoid division by zero or log of zero if any value is 0
    # In practice, AUCs are rarely exactly 0, but we clip for safety
    series = np.clip(series, 1e-6, 1.0)
    total = np.sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def compute_bias_metrics(df, pred_col, label_col, identity_cols):
    """
    Computes per-identity bias metrics: Subgroup AUC, BPSN AUC, BNSP AUC.

    Args:
        df: DataFrame containing targets, identity columns, and predictions.
        pred_col: Name of the prediction column.
        label_col: Name of the target column.
        identity_cols: List of identity column names.

    Returns:
        DataFrame with columns ['subgroup', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    """
    # Work on a copy to avoid side effects
    eval_df = df.copy()

    # Binarize target (Task: target >= 0.5 is positive)
    eval_df["is_toxic"] = (eval_df[label_col] >= 0.5).astype(int)

    records = []

    for identity in identity_cols:
        # Binarize identity (Standard practice: >= 0.5 is a mention)
        eval_df["is_identity"] = (eval_df[identity] >= 0.5).astype(int)

        # Define boolean masks
        mask_identity = eval_df["is_identity"] == 1
        mask_toxic = eval_df["is_toxic"] == 1
        mask_nontoxic = eval_df["is_toxic"] == 0

        # 1. Subgroup AUC
        # Restrict to examples mentioning the identity
        subgroup_df = eval_df[mask_identity]
        subgroup_auc = calculate_auc(subgroup_df["is_toxic"], subgroup_df[pred_col])

        # 2. BPSN AUC (Background Positive, Subgroup Negative)
        # Background Positive: Toxic & No Identity
        # Subgroup Negative: Non-Toxic & Identity
        # A low value means the model confuses non-toxic identity mentions with toxic comments.
        bpsn_mask = (mask_toxic & (~mask_identity)) | (mask_nontoxic & mask_identity)
        bpsn_df = eval_df[bpsn_mask]
        bpsn_auc = calculate_auc(bpsn_df["is_toxic"], bpsn_df[pred_col])

        # 3. BNSP AUC (Background Negative, Subgroup Positive)
        # Background Negative: Non-Toxic & No Identity
        # Subgroup Positive: Toxic & Identity
        # A low value means the model confuses toxic identity mentions with non-toxic comments.
        bnsp_mask = (mask_nontoxic & (~mask_identity)) | (mask_toxic & mask_identity)
        bnsp_df = eval_df[bnsp_mask]
        bnsp_auc = calculate_auc(bnsp_df["is_toxic"], bnsp_df[pred_col])

        records.append(
            {
                "subgroup": identity,
                "subgroup_auc": subgroup_auc,
                "bpsn_auc": bpsn_auc,
                "bnsp_auc": bnsp_auc,
            }
        )

    return pd.DataFrame(records)


def calculate_final_score(df, pred_col="prediction", label_col="target"):
    """
    Calculates the final weighted score combining Overall AUC and Bias AUCs.

    Formula: score = 0.25*Overall_AUC + 0.25*Mp(Subgroup) + 0.25*Mp(BPSN) + 0.25*Mp(BNSP)

    Args:
        df: DataFrame containing targets, identity columns, and predictions.
        pred_col: Name of the prediction column.
        label_col: Name of the target column.

    Returns:
        Dictionary containing the final score and component metrics.
    """
    # 1. Overall AUC
    y_true = (df[label_col] >= 0.5).astype(int)
    overall_auc = calculate_auc(y_true, df[pred_col])

    # 2. Compute Bias Metrics per Identity
    bias_metrics_df = compute_bias_metrics(
        df, pred_col, label_col, Config.IDENTITY_COLUMNS
    )

    # 3. Calculate Generalized Means (p = -5)
    p_value = -5

    # We extract the series for each metric type
    mean_subgroup_auc = power_mean(bias_metrics_df["subgroup_auc"], p_value)
    mean_bpsn_auc = power_mean(bias_metrics_df["bpsn_auc"], p_value)
    mean_bnsp_auc = power_mean(bias_metrics_df["bnsp_auc"], p_value)

    # 4. Weighted Sum
    # Weights are 0.25 for all 4 components
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * mean_subgroup_auc)
        + (0.25 * mean_bpsn_auc)
        + (0.25 * mean_bnsp_auc)
    )

    return {
        "score": final_score,
        "overall_auc": overall_auc,
        "mean_subgroup_auc": mean_subgroup_auc,
        "mean_bpsn_auc": mean_bpsn_auc,
        "mean_bnsp_auc": mean_bnsp_auc,
        "per_identity_metrics": bias_metrics_df,
    }
