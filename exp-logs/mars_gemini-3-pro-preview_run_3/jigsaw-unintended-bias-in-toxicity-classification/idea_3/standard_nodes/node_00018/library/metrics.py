import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def compute_auc(y_true, y_pred):
    """
    Safely compute ROC AUC. Returns NaN if only one class is present in y_true.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def power_mean(series, p):
    """
    Computes the generalized mean (power mean) of a series.
    """
    # Filter out NaNs (e.g., if a subgroup didn't exist in the batch/split)
    series = np.array(series)
    series = series[~np.isnan(series)]

    if len(series) == 0:
        return 0.0

    total = np.sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def calculate_bias_metrics(val_df, y_pred):
    """
    Calculates the bias metrics (Subgroup AUC, BPSN, BNSP) for each identity.

    Args:
        val_df (pd.DataFrame): DataFrame containing 'target' and identity columns.
        y_pred (np.array): Predicted probabilities for the positive class.

    Returns:
        pd.DataFrame: A DataFrame where index is identity name and columns are the 3 bias AUCs.
    """
    # Ensure inputs are aligned
    y_true = val_df[Config.TARGET_COL].values

    # Convert target and identities to boolean for subsetting logic
    # The competition specifies target >= 0.5 is positive
    y_true_bool = (y_true >= 0.5).astype(bool)

    records = []

    for identity_col in Config.IDENTITY_COLUMNS:
        if identity_col not in val_df.columns:
            continue

        # Identity is present if value >= 0.5
        identity_mask = (val_df[identity_col].values >= 0.5).astype(bool)

        # 1. Subgroup AUC
        # Restrict to examples mentioning the identity
        subgroup_mask = identity_mask
        subgroup_auc = compute_auc(y_true_bool[subgroup_mask], y_pred[subgroup_mask])

        # 2. BPSN AUC (Background Positive, Subgroup Negative)
        # Non-toxic examples that mention identity (False Positives candidates)
        # AND Toxic examples that do not mention identity (True Positives)
        # We want to distinguish these.
        bpsn_mask = (identity_mask & ~y_true_bool) | (~identity_mask & y_true_bool)
        bpsn_auc = compute_auc(y_true_bool[bpsn_mask], y_pred[bpsn_mask])

        # 3. BNSP AUC (Background Negative, Subgroup Positive)
        # Toxic examples that mention identity (False Negative candidates)
        # AND Non-toxic examples that do not mention identity (True Negatives)
        bnsp_mask = (identity_mask & y_true_bool) | (~identity_mask & ~y_true_bool)
        bnsp_auc = compute_auc(y_true_bool[bnsp_mask], y_pred[bnsp_mask])

        records.append(
            {
                "subgroup": identity_col,
                "subgroup_auc": subgroup_auc,
                "bpsn_auc": bpsn_auc,
                "bnsp_auc": bnsp_auc,
            }
        )

    return pd.DataFrame(records).set_index("subgroup")


def compute_final_score(val_df, y_pred):
    """
    Computes the final competition score combining Overall AUC and Bias Metrics.

    Args:
        val_df (pd.DataFrame): Validation DataFrame with targets and identities.
        y_pred (np.array): Model predictions.

    Returns:
        float: The final weighted score.
        dict: Dictionary containing the breakdown of metrics.
    """
    # 1. Overall AUC
    y_true_bool = (val_df[Config.TARGET_COL].values >= 0.5).astype(int)
    overall_auc = roc_auc_score(y_true_bool, y_pred)

    # 2. Bias Metrics per subgroup
    bias_df = calculate_bias_metrics(val_df, y_pred)

    # 3. Generalized Means (p = -5)
    # We use -5 to severely penalize low scores
    p_value = -5
    subgroup_mean = power_mean(bias_df["subgroup_auc"], p_value)
    bpsn_mean = power_mean(bias_df["bpsn_auc"], p_value)
    bnsp_mean = power_mean(bias_df["bnsp_auc"], p_value)

    # 4. Final Weighted Score
    # Formula: w0*Overall + w1*Subgroup_Mean + w2*BPSN_Mean + w3*BNSP_Mean
    # All weights are 0.25
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * subgroup_mean)
        + (0.25 * bpsn_mean)
        + (0.25 * bnsp_mean)
    )

    metrics_summary = {
        "final_score": final_score,
        "overall_auc": overall_auc,
        "subgroup_mean": subgroup_mean,
        "bpsn_mean": bpsn_mean,
        "bnsp_mean": bnsp_mean,
        "bias_breakdown": bias_df.to_dict(orient="index"),
    }

    return final_score, metrics_summary
