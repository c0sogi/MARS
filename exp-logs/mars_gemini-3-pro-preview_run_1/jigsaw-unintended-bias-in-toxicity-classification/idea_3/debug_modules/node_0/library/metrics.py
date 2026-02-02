import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import print_metric


def calculate_auc(y_true, y_pred):
    """
    Calculates ROC AUC score. Returns np.nan if the subset contains only one class.
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

    Args:
        series (list or np.array): List of metric values.
        p (int): The power parameter (e.g., -5).

    Returns:
        float: The generalized mean.
    """
    # Filter out NaNs
    clean_series = np.array([x for x in series if not np.isnan(x)])
    if len(clean_series) == 0:
        return np.nan

    total = np.sum(np.power(clean_series, p))
    return np.power(total / len(clean_series), 1 / p)


def compute_bias_metrics(df, identity_col, label_col, pred_col):
    """
    Computes the three bias metrics (Subgroup AUC, BPSN AUC, BNSP AUC)
    for a single identity subgroup.

    Args:
        df (pd.DataFrame): Dataframe containing labels, predictions, and identity info.
        identity_col (str): Name of the identity column.
        label_col (str): Name of the binary target column.
        pred_col (str): Name of the prediction column.

    Returns:
        tuple: (subgroup_auc, bpsn_auc, bnsp_auc)
    """
    # Convert to boolean vectors for fast masking
    # Task description: target >= 0.5 is positive class.
    # Standard practice: identity >= 0.5 is considered a mention.
    y_true = df[label_col].values
    y_pred = df[pred_col].values

    # Boolean masks
    # Note: df[label_col] is already binary (0 or 1) based on how we prepare data,
    # but we ensure it here.
    is_toxic = (df[label_col] >= 0.5).values
    is_identity = (df[identity_col] >= 0.5).values

    # 1. Subgroup AUC
    # Restrict to examples that mention the identity
    mask_subgroup = is_identity
    subgroup_auc = calculate_auc(y_true[mask_subgroup], y_pred[mask_subgroup])

    # 2. BPSN AUC (Background Positive, Subgroup Negative)
    # Non-toxic examples that mention identity (Subgroup Negative)
    # AND Toxic examples that do not (Background Positive)
    subgroup_negative = (~is_toxic) & is_identity
    background_positive = is_toxic & (~is_identity)
    mask_bpsn = subgroup_negative | background_positive
    bpsn_auc = calculate_auc(y_true[mask_bpsn], y_pred[mask_bpsn])

    # 3. BNSP AUC (Background Negative, Subgroup Positive)
    # Toxic examples that mention identity (Subgroup Positive)
    # AND Non-toxic examples that do not (Background Negative)
    subgroup_positive = is_toxic & is_identity
    background_negative = (~is_toxic) & (~is_identity)
    mask_bnsp = subgroup_positive | background_negative
    bnsp_auc = calculate_auc(y_true[mask_bnsp], y_pred[mask_bnsp])

    return subgroup_auc, bpsn_auc, bnsp_auc


def compute_final_metric(df, prediction_col, label_col="target", verbose=True):
    """
    Computes the final weighted score combining Overall AUC and Bias AUCs.

    Args:
        df (pd.DataFrame): Validation dataframe with targets, predictions, and identities.
        prediction_col (str): Column name for model predictions.
        label_col (str): Column name for ground truth targets.
        verbose (bool): Whether to print the metrics.

    Returns:
        float: The final calculated score.
    """
    # Ensure binary target for evaluation
    # The dataset 'target' is fractional. We create a temporary binary column for AUC calc.
    binary_target_col = "binary_target_eval"
    df[binary_target_col] = (df[label_col] >= 0.5).astype(int)

    # 1. Overall AUC
    overall_auc = roc_auc_score(df[binary_target_col], df[prediction_col])

    # 2. Bias AUCs per subgroup
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    for identity in Config.IDENTITY_COLUMNS:
        sg_auc, bpsn_auc, bnsp_auc = compute_bias_metrics(
            df, identity, binary_target_col, prediction_col
        )
        subgroup_aucs.append(sg_auc)
        bpsn_aucs.append(bpsn_auc)
        bnsp_aucs.append(bnsp_auc)

    # 3. Generalized Means (p = -5)
    # We ignore NaNs (which happen if a subgroup is empty or has only 1 class in the subset)
    avg_subgroup_auc = power_mean(subgroup_aucs, -5)
    avg_bpsn_auc = power_mean(bpsn_aucs, -5)
    avg_bnsp_auc = power_mean(bnsp_aucs, -5)

    # 4. Final Weighted Score
    # score = w0 * Overall + w1 * Subgroup + w2 * BPSN + w3 * BNSP
    # All weights = 0.25
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * avg_subgroup_auc)
        + (0.25 * avg_bpsn_auc)
        + (0.25 * avg_bnsp_auc)
    )

    if verbose:
        print("=" * 30)
        print("METRICS REPORT")
        print("=" * 30)
        print_metric("Overall AUC", overall_auc)
        print_metric("Bias - Subgroup AUC (Mean)", avg_subgroup_auc)
        print_metric("Bias - BPSN AUC (Mean)", avg_bpsn_auc)
        print_metric("Bias - BNSP AUC (Mean)", avg_bnsp_auc)
        print("-" * 30)
        print_metric("FINAL SCORE", final_score)
        print("=" * 30)

    return final_score
