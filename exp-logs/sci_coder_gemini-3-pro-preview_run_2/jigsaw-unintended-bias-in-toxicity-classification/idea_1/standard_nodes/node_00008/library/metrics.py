import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import IDENTITY_COLUMNS


def compute_auc(y_true, y_pred):
    """
    Safely compute ROC-AUC score.
    Returns np.nan if the set of labels contains only one class.
    """
    try:
        # Check if both classes are present
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def power_mean(series, p):
    """
    Calculate the generalized mean (power mean) of a series.
    M_p(x) = (1/N * sum(x^p))^(1/p)
    Ignores NaN values.
    """
    # Drop NaNs and convert to numpy array
    s = np.array(series.dropna())
    if len(s) == 0:
        return np.nan

    # Clip values to avoid overflow/underflow with large negative powers
    # AUC is usually in [0, 1]. 0.0 would cause inf with negative p.
    s = np.clip(s, 1e-6, 1.0)

    total = np.sum(np.power(s, p))
    return np.power(total / len(s), 1 / p)


def compute_subgroup_auc(df, subgroup, label_col, pred_col):
    """
    Computes the AUC for the subset of the data where the subgroup identity is present.
    """
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label_col], subgroup_examples[pred_col])


def compute_bpsn_auc(df, subgroup, label_col, pred_col):
    """
    Computes the BPSN (Background Positive, Subgroup Negative) AUC.
    Subset: (Non-toxic & Subgroup) U (Toxic & No-Subgroup)
    Target: 1 if (Toxic & No-Subgroup), 0 if (Non-toxic & Subgroup)
    """
    # Subgroup Negative: Identity is present, but comment is Non-Toxic (Target=0)
    subgroup_negative = df[df[subgroup] & ~df[label_col]]

    # Background Positive: Identity is NOT present, but comment is Toxic (Target=1)
    background_positive = df[~df[subgroup] & df[label_col]]

    # Combine subsets
    examples = pd.concat([subgroup_negative, background_positive])

    # The label_col already has the correct binary values:
    # subgroup_negative -> 0
    # background_positive -> 1
    return compute_auc(examples[label_col], examples[pred_col])


def compute_bnsp_auc(df, subgroup, label_col, pred_col):
    """
    Computes the BNSP (Background Negative, Subgroup Positive) AUC.
    Subset: (Toxic & Subgroup) U (Non-toxic & No-Subgroup)
    Target: 1 if (Toxic & Subgroup), 0 if (Non-toxic & No-Subgroup)
    """
    # Subgroup Positive: Identity is present, and comment is Toxic (Target=1)
    subgroup_positive = df[df[subgroup] & df[label_col]]

    # Background Negative: Identity is NOT present, and comment is Non-Toxic (Target=0)
    background_negative = df[~df[subgroup] & ~df[label_col]]

    # Combine subsets
    examples = pd.concat([subgroup_positive, background_negative])

    # The label_col already has the correct binary values:
    # subgroup_positive -> 1
    # background_negative -> 0
    return compute_auc(examples[label_col], examples[pred_col])


def compute_final_metric(df, label_col, pred_col, identity_columns=IDENTITY_COLUMNS):
    """
    Computes the final competition metric, which is a weighted average of the Overall AUC
    and three bias-related generalized means (Subgroup, BPSN, BNSP).

    Args:
        df (pd.DataFrame): DataFrame containing labels, predictions, and identity columns.
        label_col (str): Name of the column containing the ground truth (fractional or binary).
        pred_col (str): Name of the column containing the predicted probabilities.
        identity_columns (list): List of identity column names to evaluate for bias.

    Returns:
        final_score (float): The calculated competition metric.
        metrics_dict (dict): A dictionary containing the breakdown of scores.
    """
    # Create a working copy to avoid modifying the original dataframe
    eval_df = df.copy()

    # Convert target to boolean (Threshold >= 0.5 as per competition rules)
    eval_df["bool_target"] = (eval_df[label_col] >= 0.5).astype(bool)

    # Convert identity columns to boolean (Threshold >= 0.5)
    # This defines whether an identity is "mentioned" or not.
    for col in identity_columns:
        eval_df[col] = (eval_df[col] >= 0.5).astype(bool)

    # 1. Calculate Overall AUC
    overall_auc = compute_auc(eval_df["bool_target"], eval_df[pred_col])

    # 2. Calculate Bias AUCs for each identity subgroup
    bias_records = []
    for subgroup in identity_columns:
        subgroup_auc = compute_subgroup_auc(eval_df, subgroup, "bool_target", pred_col)
        bpsn_auc = compute_bpsn_auc(eval_df, subgroup, "bool_target", pred_col)
        bnsp_auc = compute_bnsp_auc(eval_df, subgroup, "bool_target", pred_col)

        bias_records.append(
            {
                "subgroup": subgroup,
                "subgroup_auc": subgroup_auc,
                "bpsn_auc": bpsn_auc,
                "bnsp_auc": bnsp_auc,
            }
        )

    bias_df = pd.DataFrame(bias_records)

    # 3. Calculate Generalized Means (Power Mean with p = -5)
    # This encourages improving the worst-performing subgroups.
    p_value = -5
    mp_subgroup = power_mean(bias_df["subgroup_auc"], p_value)
    mp_bpsn = power_mean(bias_df["bpsn_auc"], p_value)
    mp_bnsp = power_mean(bias_df["bnsp_auc"], p_value)

    # 4. Calculate Final Weighted Score
    # Weights are 0.25 for each of the 4 components
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * mp_subgroup)
        + (0.25 * mp_bpsn)
        + (0.25 * mp_bnsp)
    )

    metrics_dict = {
        "final_score": final_score,
        "overall_auc": overall_auc,
        "mp_subgroup_auc": mp_subgroup,
        "mp_bpsn_auc": mp_bpsn,
        "mp_bnsp_auc": mp_bnsp,
        "per_subgroup": bias_df.to_dict(orient="records"),
    }

    return final_score, metrics_dict
