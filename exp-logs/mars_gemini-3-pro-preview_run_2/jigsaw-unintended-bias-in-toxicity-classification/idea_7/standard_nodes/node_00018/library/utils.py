import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (cuda or cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_auc(y_true, y_pred):
    """
    Safe computation of ROC AUC. Returns 0.5 if only one class is present.
    """
    try:
        if len(np.unique(y_true)) == 1:
            return 0.5
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def power_mean(series, p):
    """
    Computes the generalized mean (power mean) of a series.
    """
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def calculate_jigsaw_metrics(
    val_df,
    prediction_col,
    target_col=Config.TARGET_COL,
    identity_columns=Config.IDENTITY_COLUMNS,
):
    """
    Calculates the competition metric:
    Score = 0.25*Overall_AUC + 0.25*Mp(Subgroup_AUC) + 0.25*Mp(BPSN_AUC) + 0.25*Mp(BNSP_AUC)
    where Mp is the power mean with p=-5.

    Args:
        val_df: DataFrame containing targets, identities, and predictions.
        prediction_col: Name of the column containing model predictions.
        target_col: Name of the target column (fractional).
        identity_columns: List of identity column names.

    Returns:
        dict: Dictionary containing the final score and breakdown of metrics.
    """
    # Convert fractional target and identities to booleans for metric calculation
    # Standard Jigsaw threshold is 0.5
    y_true_bool = (val_df[target_col] >= 0.5).astype(bool)
    y_pred = val_df[prediction_col]

    # 1. Overall AUC
    overall_auc = compute_auc(y_true_bool, y_pred)

    # Initialize lists to store per-identity scores
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    for identity in identity_columns:
        if identity not in val_df.columns:
            continue

        ident_bool = (val_df[identity] >= 0.5).astype(bool)

        # Subgroup AUC: Calculate AUC on examples that mention the identity
        subgroup_mask = ident_bool
        if subgroup_mask.sum() > 0:
            subgroup_auc = compute_auc(
                y_true_bool[subgroup_mask], y_pred[subgroup_mask]
            )
        else:
            subgroup_auc = 0.5
        subgroup_aucs.append(subgroup_auc)

        # BPSN AUC: Background Positive, Subgroup Negative
        # Non-toxic examples that mention identity AND Toxic examples that do not
        # (y_true=0 & ident=1) | (y_true=1 & ident=0)
        bpsn_mask = (~y_true_bool & ident_bool) | (y_true_bool & ~ident_bool)
        if bpsn_mask.sum() > 0:
            bpsn_auc = compute_auc(y_true_bool[bpsn_mask], y_pred[bpsn_mask])
        else:
            bpsn_auc = 0.5
        bpsn_aucs.append(bpsn_auc)

        # BNSP AUC: Background Negative, Subgroup Positive
        # Toxic examples that mention identity AND Non-toxic examples that do not
        # (y_true=1 & ident=1) | (y_true=0 & ident=0)
        bnsp_mask = (y_true_bool & ident_bool) | (~y_true_bool & ~ident_bool)
        if bnsp_mask.sum() > 0:
            bnsp_auc = compute_auc(y_true_bool[bnsp_mask], y_pred[bnsp_mask])
        else:
            bnsp_auc = 0.5
        bnsp_aucs.append(bnsp_auc)

    # Compute Generalized Means (p = -5)
    # We use a small epsilon or handle cases where AUC might be 0 (though unlikely for AUC)
    # AUC is typically [0, 1]. If AUC is 0, power mean might fail.
    # However, standard AUC won't be exactly 0 unless perfectly wrong.

    p = -5
    gen_mean_subgroup = power_mean(subgroup_aucs, p)
    gen_mean_bpsn = power_mean(bpsn_aucs, p)
    gen_mean_bnsp = power_mean(bnsp_aucs, p)

    # Final Score
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * gen_mean_subgroup)
        + (0.25 * gen_mean_bpsn)
        + (0.25 * gen_mean_bnsp)
    )

    return {
        "score": final_score,
        "overall_auc": overall_auc,
        "subgroup_auc_mean": gen_mean_subgroup,
        "bpsn_auc_mean": gen_mean_bpsn,
        "bnsp_auc_mean": gen_mean_bnsp,
    }
