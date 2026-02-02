import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def compute_auc(y_true, y_pred):
    """
    Safely computes ROC-AUC. Returns 0.5 if only one class is present in y_true.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def compute_subgroup_auc(df, identity_col, label_col, pred_col):
    """
    Computes AUC on the subset of examples where the identity is mentioned.
    """
    mask = df[identity_col] == 1
    sub_df = df[mask]
    return compute_auc(sub_df[label_col], sub_df[pred_col])


def compute_bpsn_auc(df, identity_col, label_col, pred_col):
    """
    Computes BPSN AUC (Background Positive, Subgroup Negative).
    Subset: (Non-toxic & Identity) U (Toxic & No-Identity)
    """
    # Non-toxic (0) and Identity (1) -> This is the "Subgroup Negative" part
    subgroup_negative = (df[label_col] == 0) & (df[identity_col] == 1)

    # Toxic (1) and No-Identity (0) -> This is the "Background Positive" part
    background_positive = (df[label_col] == 1) & (df[identity_col] == 0)

    mask = subgroup_negative | background_positive
    sub_df = df[mask]
    return compute_auc(sub_df[label_col], sub_df[pred_col])


def compute_bnsp_auc(df, identity_col, label_col, pred_col):
    """
    Computes BNSP AUC (Background Negative, Subgroup Positive).
    Subset: (Toxic & Identity) U (Non-toxic & No-Identity)
    """
    # Toxic (1) and Identity (1) -> This is the "Subgroup Positive" part
    subgroup_positive = (df[label_col] == 1) & (df[identity_col] == 1)

    # Non-toxic (0) and No-Identity (0) -> This is the "Background Negative" part
    background_negative = (df[label_col] == 0) & (df[identity_col] == 0)

    mask = subgroup_positive | background_negative
    sub_df = df[mask]
    return compute_auc(sub_df[label_col], sub_df[pred_col])


def calculate_generalized_mean(scores, p=-5):
    """
    Calculates the generalized mean (power mean) of a list of scores.
    """
    if not scores:
        return 0.0
    scores = np.array(scores)
    # Avoid division by zero or log of zero issues if scores are exactly 0
    scores = np.clip(scores, 1e-6, 1.0)
    mean = np.mean(np.power(scores, p))
    return np.power(mean, 1.0 / p)


def calculate_final_score(val_df: pd.DataFrame, predictions: np.ndarray):
    """
    Calculates the final Jigsaw competition metric.

    Args:
        val_df: Validation DataFrame containing 'target' and identity columns.
        predictions: Predicted probabilities for the positive class.

    Returns:
        final_score: The weighted composite score.
        metrics_dict: A dictionary containing detailed sub-metrics.
    """
    # Create a working copy to avoid modifying the original dataframe
    df = val_df.copy()

    # Add predictions
    pred_col = "prediction"
    df[pred_col] = predictions

    # Binarize target and identity columns (Threshold >= 0.5)
    label_col = "binary_target"
    if label_col not in df.columns:
        df[label_col] = (df["target"] >= 0.5).astype(int)

    # Ensure identity columns are binary for the metric calculation
    # We use temporary column names to avoid overwriting original continuous data if needed elsewhere
    bin_identity_cols = []
    for col in Config.IDENTITY_COLS:
        bin_col = f"{col}_bin"
        # Fill NaNs with 0 (assuming NaN means identity not mentioned)
        df[bin_col] = (df[col].fillna(0.0) >= 0.5).astype(int)
        bin_identity_cols.append(bin_col)

    # 1. Overall AUC
    overall_auc = compute_auc(df[label_col], df[pred_col])

    # 2. Calculate Bias AUCs per identity
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    metrics_dict = {"overall_auc": overall_auc}

    for i, ident_col in enumerate(Config.IDENTITY_COLS):
        bin_ident_col = bin_identity_cols[i]

        # Subgroup AUC
        s_auc = compute_subgroup_auc(df, bin_ident_col, label_col, pred_col)
        subgroup_aucs.append(s_auc)
        metrics_dict[f"{ident_col}_subgroup_auc"] = s_auc

        # BPSN AUC
        bpsn_auc = compute_bpsn_auc(df, bin_ident_col, label_col, pred_col)
        bpsn_aucs.append(bpsn_auc)
        metrics_dict[f"{ident_col}_bpsn_auc"] = bpsn_auc

        # BNSP AUC
        bnsp_auc = compute_bnsp_auc(df, bin_ident_col, label_col, pred_col)
        bnsp_aucs.append(bnsp_auc)
        metrics_dict[f"{ident_col}_bnsp_auc"] = bnsp_auc

    # 3. Calculate Generalized Means (p = -5)
    # The task description specifies p=-5 to encourage improving the worst performing subgroups
    gen_mean_subgroup = calculate_generalized_mean(subgroup_aucs, p=-5)
    gen_mean_bpsn = calculate_generalized_mean(bpsn_aucs, p=-5)
    gen_mean_bnsp = calculate_generalized_mean(bnsp_aucs, p=-5)

    metrics_dict["gen_mean_subgroup"] = gen_mean_subgroup
    metrics_dict["gen_mean_bpsn"] = gen_mean_bpsn
    metrics_dict["gen_mean_bnsp"] = gen_mean_bnsp

    # 4. Final Weighted Score
    # score = w0*Overall + w1*Subgroup + w2*BPSN + w3*BNSP
    # All weights = 0.25
    final_score = (
        0.25 * overall_auc
        + 0.25 * gen_mean_subgroup
        + 0.25 * gen_mean_bpsn
        + 0.25 * gen_mean_bnsp
    )

    metrics_dict["final_score"] = final_score

    return final_score, metrics_dict
