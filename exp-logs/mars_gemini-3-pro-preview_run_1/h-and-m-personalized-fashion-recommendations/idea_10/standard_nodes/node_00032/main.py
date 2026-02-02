import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, Timer
from library.tessc_model import TESSCRecommender
from library.evaluation import calculate_map12


def calculate_per_user_ap(predictions, ground_truth, k=12):
    """
    Calculates Average Precision (AP) per user for failure analysis.
    Returns a DataFrame with customer_id and ap_score.
    """
    # Prepare Ground Truth
    if "article_id" in ground_truth.columns:
        sample_val = (
            ground_truth["article_id"].iloc[0] if not ground_truth.empty else None
        )
        if isinstance(sample_val, (list, np.ndarray, tuple)):
            truth_grouped = ground_truth.copy().rename(columns={"article_id": "actual"})
        else:
            truth_grouped = (
                ground_truth.groupby("customer_id")["article_id"]
                .apply(list)
                .reset_index()
                .rename(columns={"article_id": "actual"})
            )
    else:
        return pd.DataFrame()

    # Prepare Predictions
    pred_df = predictions[["customer_id", "prediction"]].copy()
    pred_df["prediction"] = pred_df["prediction"].fillna("")
    pred_df["predicted"] = pred_df["prediction"].str.split()

    # Merge
    merged = pd.merge(truth_grouped, pred_df, on="customer_id", how="left")

    user_scores = []
    customer_ids = []

    actuals = merged["actual"].values
    predicteds = merged["predicted"].values
    c_ids = merged["customer_id"].values

    for cid, actual, predicted in zip(c_ids, actuals, predicteds):
        if not isinstance(predicted, list):
            predicted = []

        predicted = predicted[:k]

        if not actual:
            user_scores.append(0.0)
            customer_ids.append(cid)
            continue

        actual_set = set(str(x) for x in actual)
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if str(p) in actual_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        denom = min(len(actual_set), k)
        ap = score / denom if denom > 0 else 0.0

        user_scores.append(ap)
        customer_ids.append(cid)

    return pd.DataFrame({"customer_id": customer_ids, "ap_score": user_scores})


def perform_failure_analysis(model, predictions, ground_truth):
    """
    Correlates model error (1 - AP) with user features derived from the interaction matrix.
    """
    print("\n[Failure Analysis] Starting...")

    # 1. Calculate Per-User AP
    ap_df = calculate_per_user_ap(predictions, ground_truth)
    ap_df["error"] = 1.0 - ap_df["ap_score"]

    # 2. Extract User Features from Model's Interaction Matrix (X)
    # We need to map customer_id -> user_idx -> row stats in X
    # model.cust_to_idx is available after fit

    # Filter ap_df to users present in the training set (to get features)
    # Users in validation but not in training are cold-start for the model (handled by trend)

    user_features = []

    # Get the sparse matrix and map
    X = model.X
    cust_to_idx = model.cust_to_idx

    # We iterate through the AP dataframe
    # This might be slow if we do it row by row for millions, but validation set is smaller (~270k users)
    # Vectorized approach:

    valid_customers = ap_df["customer_id"].values
    valid_indices = []
    mask = []

    for cid in valid_customers:
        if cid in cust_to_idx:
            valid_indices.append(cust_to_idx[cid])
            mask.append(True)
        else:
            valid_indices.append(-1)  # Placeholder
            mask.append(False)

    ap_subset = ap_df[mask].copy()
    valid_indices = [i for i in valid_indices if i != -1]

    if not valid_indices:
        print(
            "[Failure Analysis] No overlap between validation users and training history. Skipping correlation."
        )
        return

    # Extract features from Sparse Matrix X for these users
    # X is CSR.
    # Feature 1: History Length (Number of unique items bought in history window) -> nnz per row
    # Feature 2: Recency/Intensity Score (Sum of weights) -> sum per row

    X_subset = X[valid_indices]

    # Count non-zeros per row
    history_len = np.diff(X_subset.indptr)

    # Sum weights per row (matrix is float32)
    # sum returns a matrix object, need to flatten
    intensity_score = np.array(X_subset.sum(axis=1)).flatten()

    ap_subset["history_length"] = history_len
    ap_subset["intensity_score"] = intensity_score

    # 3. Compute Correlations
    corr_len = ap_subset["error"].corr(ap_subset["history_length"])
    corr_int = ap_subset["error"].corr(ap_subset["intensity_score"])

    print("-" * 40)
    print(f"Failure Analysis Report (N={len(ap_subset)} users)")
    print(f"Correlation (Error vs History Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Intensity/Recency): {corr_int:.4f}")
    print("-" * 40)

    # Interpretation
    if abs(corr_len) > 0.1:
        print(
            f"Observation: Error is {'positively' if corr_len > 0 else 'negatively'} correlated with history length."
        )
    if abs(corr_int) > 0.1:
        print(
            f"Observation: Error is {'positively' if corr_int > 0 else 'negatively'} correlated with recency intensity."
        )


def main():
    # 1. Setup
    set_seed(Config.SEED)

    print("=" * 50)
    print("Running TESSC Pipeline")
    print("=" * 50)

    # ---------------------------------------------------------
    # Phase 1: Validation
    # ---------------------------------------------------------
    print("\n--- Phase 1: Validation ---")

    # Initialize Model
    val_model = TESSCRecommender()

    # Fit on Training Split (Start -> T-7 days)
    # load_cached_data=True allows using artifacts if they exist in ./working/idea_10
    val_model.fit(use_validation=True, load_cached_data=True)

    # Identify Validation Users
    if val_model.val_df is None:
        raise RuntimeError(
            "Validation DataFrame is missing after fit(use_validation=True)."
        )

    val_customers = val_model.val_df["customer_id"].unique().tolist()
    print(f"Predicting for {len(val_customers)} validation customers...")

    # Generate Predictions
    val_preds = val_model.predict(val_customers)

    # Calculate Metric
    # Note: calculate_map12 expects ground truth df and prediction df
    map_score = calculate_map12(val_preds, val_model.val_df)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {map_score}")

    # Failure Analysis
    perform_failure_analysis(val_model, val_preds, val_model.val_df)

    # Clean up to save memory
    del val_model, val_preds, val_customers
    import gc

    gc.collect()

    # ---------------------------------------------------------
    # Phase 2: Submission
    # ---------------------------------------------------------
    print("\n--- Phase 2: Submission ---")

    THRESHOLD = 0.0265060791

    if map_score > THRESHOLD:
        print(
            f"Validation score ({map_score:.6f}) exceeds threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Initialize Model for Full Training
        full_model = TESSCRecommender()

        # Fit on Full Dataset (Start -> T_max)
        full_model.fit(use_validation=False, load_cached_data=True)

        # Generate Submission
        # This method internally loads test customers and saves to Config.SUBMISSION_PATH
        full_model.generate_submission()

    else:
        print(
            f"Validation score ({map_score:.6f}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
