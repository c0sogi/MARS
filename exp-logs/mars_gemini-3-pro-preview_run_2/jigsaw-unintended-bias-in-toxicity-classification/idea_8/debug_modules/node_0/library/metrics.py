import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def generalized_mean(scores, p=-5):
    """
    Calculates the generalized mean (power mean) of a list of scores.

    Formula: M_p(x) = (1/N * sum(x_i^p))^(1/p)

    Args:
        scores (list or np.array): List of AUC scores.
        p (int or float): The power parameter (default -5).

    Returns:
        float: The generalized mean.
    """
    scores = np.array(scores)

    # Filter out NaNs (e.g., if a subgroup didn't exist in the batch)
    scores = scores[~np.isnan(scores)]

    if len(scores) == 0:
        return 0.0

    # Clip scores to avoid numerical errors with negative powers (e.g., 0^-5)
    # AUCs are typically in [0, 1], but 0 causes div-by-zero for p < 0.
    scores = np.clip(scores, 1e-6, 1.0)

    mean_pow = np.mean(np.power(scores, p))
    return np.power(mean_pow, 1.0 / p)


def calculate_roc_auc(y_true, y_pred):
    """
    Safe calculation of ROC AUC. Returns np.nan if only one class is present.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for single class
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_pred)


def compute_bias_metrics(df, target_col, pred_col, identity_columns=None):
    """
    Computes the complete set of Jigsaw bias metrics.

    Args:
        df (pd.DataFrame): DataFrame containing targets, predictions, and identity columns.
        target_col (str): Name of the target column.
        pred_col (str): Name of the prediction column.
        identity_columns (list): List of identity column names.

    Returns:
        dict: A dictionary containing:
            - overall_auc
            - bias_auc_score (the weighted final metric)
            - mp_subgroup_auc (Generalized Mean of Subgroup AUCs)
            - mp_bpsn_auc (Generalized Mean of BPSN AUCs)
            - mp_bnsp_auc (Generalized Mean of BNSP AUCs)
            - per_identity_metrics (dict of detailed scores)
    """
    if identity_columns is None:
        identity_columns = Config.IDENTITY_COLUMNS

    # 1. Preprocessing & Stability
    # Clip predictions to prevent numerical instability
    df = df.copy()
    df[pred_col] = np.clip(df[pred_col], 1e-6, 1.0 - 1e-6)

    # Convert continuous targets/identities to boolean for subsetting
    # Note: For the actual AUC calculation, we use the boolean target as y_true
    # consistent with the competition evaluation (target >= 0.5 is positive).
    y_true_bool = (df[target_col] >= 0.5).astype(int)
    y_pred = df[pred_col]

    # 2. Overall AUC
    overall_auc = calculate_roc_auc(y_true_bool, y_pred)
    if np.isnan(overall_auc):
        overall_auc = 0.5  # Fallback

    # 3. Per-Identity Bias AUCs
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    per_identity_metrics = {}

    for identity in identity_columns:
        if identity not in df.columns:
            continue

        # Masks
        # Identity is present if value >= 0.5
        id_mask = df[identity] >= 0.5

        # Toxic: target >= 0.5
        toxic_mask = y_true_bool == 1
        nontoxic_mask = y_true_bool == 0

        # --- Subgroup AUC ---
        # Restrict to examples mentioning the identity
        mask_sub = id_mask
        auc_sub = calculate_roc_auc(y_true_bool[mask_sub], y_pred[mask_sub])

        # --- BPSN AUC (Background Positive, Subgroup Negative) ---
        # Non-toxic examples mentioning identity (Background Positive -> Label 0 in this subset logic?)
        # Wait, definitions:
        # BPSN: Non-toxic examples that mention identity AND Toxic examples that do not.
        # "Background Positive" usually refers to the class that is 0 in the main task but confusingly high.
        # In Jigsaw metric:
        #   Set: (Identity=1 & Toxic=0) U (Identity=0 & Toxic=1)
        #   We want to distinguish these.
        #   Ideally, (Identity=1 & Toxic=0) should be low, (Identity=0 & Toxic=1) should be high.
        #   So y_true remains the Toxicity label.
        mask_bpsn = (id_mask & nontoxic_mask) | (~id_mask & toxic_mask)
        auc_bpsn = calculate_roc_auc(y_true_bool[mask_bpsn], y_pred[mask_bpsn])

        # --- BNSP AUC (Background Negative, Subgroup Positive) ---
        # Toxic examples mentioning identity AND Non-toxic examples that do not.
        # Set: (Identity=1 & Toxic=1) U (Identity=0 & Toxic=0)
        mask_bnsp = (id_mask & toxic_mask) | (~id_mask & nontoxic_mask)
        auc_bnsp = calculate_roc_auc(y_true_bool[mask_bnsp], y_pred[mask_bnsp])

        # Store
        subgroup_aucs.append(auc_sub)
        bpsn_aucs.append(auc_bpsn)
        bnsp_aucs.append(auc_bnsp)

        per_identity_metrics[identity] = {
            "subgroup_auc": auc_sub,
            "bpsn_auc": auc_bpsn,
            "bnsp_auc": auc_bnsp,
        }

    # 4. Generalized Means
    mp_subgroup = generalized_mean(subgroup_aucs, p=-5)
    mp_bpsn = generalized_mean(bpsn_aucs, p=-5)
    mp_bnsp = generalized_mean(bnsp_aucs, p=-5)

    # 5. Final Score
    # score = 0.25*Overall + 0.25*Mp(Subgroup) + 0.25*Mp(BPSN) + 0.25*Mp(BNSP)
    final_score = (
        (0.25 * overall_auc)
        + (0.25 * mp_subgroup)
        + (0.25 * mp_bpsn)
        + (0.25 * mp_bnsp)
    )

    return {
        "score": final_score,
        "overall_auc": overall_auc,
        "mp_subgroup_auc": mp_subgroup,
        "mp_bpsn_auc": mp_bpsn,
        "mp_bnsp_auc": mp_bnsp,
        "per_identity_metrics": per_identity_metrics,
    }
