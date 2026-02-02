import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def calculate_overall_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the overall ROC-AUC for the full evaluation set.

    Args:
        y_true: Binary target labels.
        y_pred: Predicted probabilities.

    Returns:
        float: The ROC-AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def calculate_generalized_mean(series: pd.Series, p: int = -5) -> float:
    """
    Calculates the generalized mean (power mean) of a series of scores.
    Formula: (1/N * sum(x^p))^(1/p)

    Args:
        series: Pandas Series of metric scores (e.g., AUCs).
        p: The power parameter. Default is -5.

    Returns:
        float: The generalized mean.
    """
    # Filter out NaNs (e.g., if a subgroup didn't exist in the val set)
    data = series.dropna()

    if len(data) == 0:
        return np.nan

    # Clip values to avoid numerical instability with negative powers near zero
    # AUCs are typically > 0.5, but we ensure strict positivity.
    data = np.clip(data, 1e-6, 1.0)

    # Compute power mean
    mean_pow = np.mean(np.power(data, p))
    return np.power(mean_pow, 1.0 / p)


def compute_bias_metrics_for_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    identity_columns: list = None,
) -> pd.DataFrame:
    """
    Computes the three specific bias AUC metrics (Subgroup, BPSN, BNSP)
    for each identity subgroup.

    Args:
        y_true: Binary target labels.
        y_pred: Predicted probabilities.
        df: DataFrame containing the identity columns corresponding to y_true/y_pred.
        identity_columns: List of identity column names. Defaults to Config.IDENTITY_COLUMNS.

    Returns:
        pd.DataFrame: A DataFrame with columns ['subgroup', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc'].
    """
    if identity_columns is None:
        identity_columns = Config.IDENTITY_COLUMNS

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = []

    for col in identity_columns:
        if col not in df.columns:
            continue

        # Determine identity presence (threshold >= 0.5 for fractional labels)
        # Using .values ensures we work with numpy arrays for boolean indexing
        ident_mask = (df[col] >= 0.5).values

        # ---------------------------
        # 1. Subgroup AUC
        # ---------------------------
        # Subset: Only examples that mention the specific identity.
        sg_mask = ident_mask

        # Calculate AUC if subset has samples and both classes
        if np.sum(sg_mask) > 0 and len(np.unique(y_true[sg_mask])) > 1:
            subgroup_auc = roc_auc_score(y_true[sg_mask], y_pred[sg_mask])
        else:
            subgroup_auc = np.nan

        # ---------------------------
        # 2. BPSN AUC (Background Positive, Subgroup Negative)
        # ---------------------------
        # Subset:
        #   - Non-toxic examples that mention the identity (y=0, id=1)
        #   - Toxic examples that do NOT mention the identity (y=1, id=0)
        bpsn_mask = ((y_true == 0) & (ident_mask)) | ((y_true == 1) & (~ident_mask))

        if np.sum(bpsn_mask) > 0 and len(np.unique(y_true[bpsn_mask])) > 1:
            bpsn_auc = roc_auc_score(y_true[bpsn_mask], y_pred[bpsn_mask])
        else:
            bpsn_auc = np.nan

        # ---------------------------
        # 3. BNSP AUC (Background Negative, Subgroup Positive)
        # ---------------------------
        # Subset:
        #   - Toxic examples that mention the identity (y=1, id=1)
        #   - Non-toxic examples that do NOT mention the identity (y=0, id=0)
        bnsp_mask = ((y_true == 1) & (ident_mask)) | ((y_true == 0) & (~ident_mask))

        if np.sum(bnsp_mask) > 0 and len(np.unique(y_true[bnsp_mask])) > 1:
            bnsp_auc = roc_auc_score(y_true[bnsp_mask], y_pred[bnsp_mask])
        else:
            bnsp_auc = np.nan

        results.append(
            {
                "subgroup": col,
                "subgroup_auc": subgroup_auc,
                "bpsn_auc": bpsn_auc,
                "bnsp_auc": bnsp_auc,
            }
        )

    return pd.DataFrame(results)


def compute_final_metric(
    y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame
) -> dict:
    """
    Computes the final competition metric, combining Overall AUC and Bias AUCs.

    Score = 0.25 * Overall_AUC +
            0.25 * PowerMean(Subgroup_AUCs) +
            0.25 * PowerMean(BPSN_AUCs) +
            0.25 * PowerMean(BNSP_AUCs)

    Args:
        y_true: Binary target labels.
        y_pred: Predicted probabilities.
        df: DataFrame containing identity columns.

    Returns:
        dict: Dictionary containing the final score and intermediate metrics.
    """
    # 1. Calculate Overall AUC
    overall_auc = calculate_overall_auc(y_true, y_pred)

    # 2. Calculate per-identity Bias AUCs
    bias_metrics_df = compute_bias_metrics_for_model(y_true, y_pred, df)

    # 3. Calculate Generalized Means (p = -5)
    p_val = -5
    subgroup_mean = calculate_generalized_mean(bias_metrics_df["subgroup_auc"], p_val)
    bpsn_mean = calculate_generalized_mean(bias_metrics_df["bpsn_auc"], p_val)
    bnsp_mean = calculate_generalized_mean(bias_metrics_df["bnsp_auc"], p_val)

    # 4. Compute Final Weighted Score
    # Weights are 0.25 for each component
    score = (
        (0.25 * overall_auc)
        + (0.25 * subgroup_mean)
        + (0.25 * bpsn_mean)
        + (0.25 * bnsp_mean)
    )

    return {
        "score": score,
        "overall_auc": overall_auc,
        "subgroup_auc_mean": subgroup_mean,
        "bpsn_auc_mean": bpsn_mean,
        "bnsp_auc_mean": bnsp_mean,
        "per_identity_metrics": bias_metrics_df,
    }
