import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def power_mean(series, p):
    """
    Computes the generalized mean (power mean) of a list of values.

    Args:
        series (list or np.array): List of metric scores.
        p (float): The power parameter (e.g., -5).

    Returns:
        float: The generalized mean.
    """
    series = np.array(series)
    # Clip values to avoid numerical instability (e.g., division by zero or log of zero)
    # AUC scores are in [0, 1]. A score of 0 with negative power results in infinity.
    series = np.clip(series, 1e-6, 1.0)

    total = np.sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def calculate_jigsaw_metrics(val_df: pd.DataFrame, y_pred: np.ndarray):
    """
    Calculates the Jigsaw competition metrics, including Overall AUC and Bias AUCs.

    Args:
        val_df (pd.DataFrame): DataFrame containing 'target' and identity columns.
        y_pred (np.ndarray): Predicted probabilities for the positive class (toxic).

    Returns:
        dict: A dictionary containing the final score and detailed component metrics.
    """
    # 1. Binarize Targets and Identities
    # The competition standardizes on a 0.5 threshold for both toxicity and identity mentions.
    y_true = (val_df[Config.TARGET_COL].values >= 0.5).astype(int)

    # Ensure predictions are a flat 1D array
    y_pred = np.array(y_pred).flatten()

    # 2. Overall AUC
    try:
        overall_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        overall_auc = 0.5

    # 3. Per-Identity Bias Metrics
    identity_cols = Config.IDENTITY_COLUMNS

    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    # Pre-compute identity booleans for efficiency
    # shape: (N_samples, N_identities)
    identities_binary = (val_df[identity_cols].values >= 0.5).astype(int)

    for i, col in enumerate(identity_cols):
        ident_mask = identities_binary[:, i] == 1

        # --- Subgroup AUC ---
        # Restrict to examples mentioning the specific identity.
        # Measures: Can we distinguish Toxic vs Non-Toxic within this group?
        mask_sub = ident_mask
        if mask_sub.sum() > 0 and len(np.unique(y_true[mask_sub])) > 1:
            score = roc_auc_score(y_true[mask_sub], y_pred[mask_sub])
        else:
            score = 0.5
        subgroup_aucs.append(score)

        # --- BPSN AUC (Background Positive, Subgroup Negative) ---
        # Background Positive: Non-toxic examples that mention the identity (y=0, id=1).
        # Subgroup Negative: Toxic examples that do NOT mention the identity (y=1, id=0).
        # Measures: Does the model confuse non-toxic identity mentions with toxic comments?
        # (i.e., does it have a high False Positive Rate for this identity?)
        mask_bpsn = ((y_true == 0) & (ident_mask)) | ((y_true == 1) & (~ident_mask))
        if mask_bpsn.sum() > 0 and len(np.unique(y_true[mask_bpsn])) > 1:
            score = roc_auc_score(y_true[mask_bpsn], y_pred[mask_bpsn])
        else:
            score = 0.5
        bpsn_aucs.append(score)

        # --- BNSP AUC (Background Negative, Subgroup Positive) ---
        # Background Negative: Non-toxic examples that do NOT mention the identity (y=0, id=0).
        # Subgroup Positive: Toxic examples that mention the identity (y=1, id=1).
        # Measures: Does the model confuse toxic identity mentions with non-toxic comments?
        # (i.e., does it have a high False Negative Rate for this identity?)
        mask_bnsp = ((y_true == 1) & (ident_mask)) | ((y_true == 0) & (~ident_mask))
        if mask_bnsp.sum() > 0 and len(np.unique(y_true[mask_bnsp])) > 1:
            score = roc_auc_score(y_true[mask_bnsp], y_pred[mask_bnsp])
        else:
            score = 0.5
        bnsp_aucs.append(score)

    # 4. Generalized Means (p = -5)
    # Used to penalize low performance on specific subgroups.
    p = -5
    gmean_subgroup = power_mean(subgroup_aucs, p)
    gmean_bpsn = power_mean(bpsn_aucs, p)
    gmean_bnsp = power_mean(bnsp_aucs, p)

    # 5. Final Weighted Score
    # Formula: score = 0.25*Overall + 0.25*Subgroup + 0.25*BPSN + 0.25*BNSP
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * gmean_subgroup)
        + (0.25 * gmean_bpsn)
        + (0.25 * gmean_bnsp)
    )

    return {
        "final_score": final_score,
        "overall_auc": overall_auc,
        "subgroup_auc": gmean_subgroup,
        "bpsn_auc": gmean_bpsn,
        "bnsp_auc": gmean_bnsp,
        "per_identity_subgroup": dict(zip(identity_cols, subgroup_aucs)),
        "per_identity_bpsn": dict(zip(identity_cols, bpsn_aucs)),
        "per_identity_bnsp": dict(zip(identity_cols, bnsp_aucs)),
    }
