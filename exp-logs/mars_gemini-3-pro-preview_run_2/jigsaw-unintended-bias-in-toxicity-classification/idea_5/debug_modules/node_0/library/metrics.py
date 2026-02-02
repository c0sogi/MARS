import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates ROC AUC score safely.
    If y_true contains only one class, returns 0.5.
    """
    try:
        # Check if both classes are present
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def power_mean(series, p):
    """
    Calculates the generalized mean (power mean) of a series of numbers.
    Formula: (1/N * sum(x^p))^(1/p)
    """
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def compute_bias_metrics_for_identity(df, identity_col, target_col, prediction_col):
    """
    Computes the three bias AUCs for a specific identity subgroup.

    Args:
        df: DataFrame containing the data.
        identity_col: Name of the identity column.
        target_col: Name of the target column (binary).
        prediction_col: Name of the prediction column (probabilities).

    Returns:
        dict: Contains 'subgroup_auc', 'bpsn_auc', 'bnsp_auc' for the identity.
    """
    # Convert to boolean for subsetting
    # Note: df[target_col] is already assumed to be boolean or 0/1 integers passed from main
    # Identity columns are continuous, threshold at 0.5
    is_identity = df[identity_col] >= 0.5
    is_toxic = df[target_col] >= 0.5

    # 1. Subgroup AUC
    # Restrict to examples that mention the identity
    subgroup_df = df[is_identity]
    subgroup_auc = calculate_roc_auc(
        subgroup_df[target_col], subgroup_df[prediction_col]
    )

    # 2. BPSN AUC (Background Positive, Subgroup Negative)
    # Non-toxic examples that mention the identity AND Toxic examples that do not
    # likely meaning model predicts higher toxicity scores than it should for non-toxic examples mentioning identity
    bpsn_mask = (~is_toxic & is_identity) | (is_toxic & ~is_identity)
    bpsn_df = df[bpsn_mask]
    bpsn_auc = calculate_roc_auc(bpsn_df[target_col], bpsn_df[prediction_col])

    # 3. BNSP AUC (Background Negative, Subgroup Positive)
    # Toxic examples that mention the identity AND Non-toxic examples that do not
    # likely meaning model predicts lower toxicity scores than it should for toxic examples mentioning identity
    bnsp_mask = (is_toxic & is_identity) | (~is_toxic & ~is_identity)
    bnsp_df = df[bnsp_mask]
    bnsp_auc = calculate_roc_auc(bnsp_df[target_col], bnsp_df[prediction_col])

    return {"subgroup_auc": subgroup_auc, "bpsn_auc": bpsn_auc, "bnsp_auc": bnsp_auc}


def calculate_final_score(df, prediction_col="prediction", target_col="target"):
    """
    Calculates the final competition score based on the formula:
    score = 0.25 * Overall_AUC + 0.25 * Mean(Subgroup_AUCs) + 0.25 * Mean(BPSN_AUCs) + 0.25 * Mean(BNSP_AUCs)

    The Means are generalized means with p = -5.

    Args:
        df (pd.DataFrame): DataFrame containing targets, predictions, and identity columns.
        prediction_col (str): Column name for model predictions.
        target_col (str): Column name for ground truth targets.

    Returns:
        float: The final weighted score.
        dict: A dictionary containing the breakdown of metrics.
    """
    # Ensure target is binary for AUC calculation (threshold 0.5)
    # We create a temporary boolean target column for calculation
    binary_target_col = "binary_target"
    df[binary_target_col] = (df[target_col] >= 0.5).astype(int)

    # 1. Overall AUC
    overall_auc = calculate_roc_auc(df[binary_target_col], df[prediction_col])

    # 2. Per-Identity Bias AUCs
    identity_scores = {"subgroup_auc": [], "bpsn_auc": [], "bnsp_auc": []}

    # Iterate over all identities defined in Config
    valid_identities = [col for col in Config.IDENTITY_COLUMNS if col in df.columns]

    if not valid_identities:
        print(
            "Warning: No identity columns found in DataFrame. Returning Overall AUC only."
        )
        return overall_auc, {"overall_auc": overall_auc}

    for identity_col in valid_identities:
        metrics = compute_bias_metrics_for_identity(
            df, identity_col, binary_target_col, prediction_col
        )
        identity_scores["subgroup_auc"].append(metrics["subgroup_auc"])
        identity_scores["bpsn_auc"].append(metrics["bpsn_auc"])
        identity_scores["bnsp_auc"].append(metrics["bnsp_auc"])

    # 3. Generalized Means (p = -5)
    p_value = -5
    subgroup_mean = power_mean(identity_scores["subgroup_auc"], p_value)
    bpsn_mean = power_mean(identity_scores["bpsn_auc"], p_value)
    bnsp_mean = power_mean(identity_scores["bnsp_auc"], p_value)

    # 4. Final Weighted Score
    # w0 = 0.25, wa = 0.25 for all three bias metrics
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * subgroup_mean)
        + (0.25 * bpsn_mean)
        + (0.25 * bnsp_mean)
    )

    # Detailed results dictionary
    results = {
        "final_score": final_score,
        "overall_auc": overall_auc,
        "subgroup_auc_mean": subgroup_mean,
        "bpsn_auc_mean": bpsn_mean,
        "bnsp_auc_mean": bnsp_mean,
        # Include per-identity scores for deeper analysis if needed
        "per_identity_subgroup": dict(
            zip(valid_identities, identity_scores["subgroup_auc"])
        ),
        "per_identity_bpsn": dict(zip(valid_identities, identity_scores["bpsn_auc"])),
        "per_identity_bnsp": dict(zip(valid_identities, identity_scores["bnsp_auc"])),
    }

    return final_score, results
