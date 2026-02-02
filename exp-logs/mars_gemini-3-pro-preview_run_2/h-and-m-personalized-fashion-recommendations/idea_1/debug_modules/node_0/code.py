import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure local library modules can be imported
sys.path.append(os.getcwd())

from library.config import WORKING_DIR
from library.data_factory import load_and_filter_data, IdEncoder
from library.model import TimeWeightedCooccurrence
from library.metrics import calculate_map12, mapk


def main():
    # 1. Setup and Reproducibility
    print("=== Starting H&M Recommendation Demo ===")
    SEED = 42
    np.random.seed(SEED)

    # 2. Demonstrate Metrics (Unit Test style)
    print("\n[1/5] Verifying Metric Logic (MAP@12)...")

    # Test Case: Perfect Match
    actual = [["101", "102"]]
    predicted = [["101", "102", "103"]]
    score = mapk(actual, predicted, k=12)
    print(f"   Perfect Match Score: {score:.4f}")
    assert score == 1.0, "Metric check failed: Expected 1.0 for perfect match"

    # Test Case: No Match
    actual = [["101"]]
    predicted = [["201", "202"]]
    score = mapk(actual, predicted, k=12)
    print(f"   No Match Score:      {score:.4f}")
    assert score == 0.0, "Metric check failed: Expected 0.0 for no match"

    # Test Case: Partial/Reordered
    # Ground truth: A, B. Pred: B, A.
    # P@1 (B): Correct? Yes (B in GT). Precision=1/1.
    # P@2 (A): Correct? Yes (A in GT). Precision=2/2.
    # AP = (1 + 1)/2 = 1.0
    actual = [["A", "B"]]
    predicted = [["B", "A"]]
    score = mapk(actual, predicted, k=12)
    print(f"   Reordered Score:     {score:.4f}")
    assert score == 1.0, "Metric check failed: Expected 1.0 for reordered hits"

    # 3. Demonstrate IdEncoder
    print("\n[2/5] Verifying IdEncoder...")
    encoder = IdEncoder()
    dummy_cust = ["c1", "c2", "c3"]
    dummy_art = ["a1", "a2", "a3", "a1"]

    # Fit (disable cache to keep this isolated)
    encoder.fit(dummy_cust, dummy_art, load_cached_data=False)

    # Transform
    c_vec = encoder.transform_customers(["c1", "c_unknown"])
    print(f"   Encoded Customers: {c_vec}")
    assert c_vec[0] != -1, "Known customer should have valid ID"
    assert c_vec[1] == -1, "Unknown customer should be -1"

    # Inverse Transform
    a_vec = encoder.transform_articles(["a1", "a2"])
    decoded = encoder.inverse_transform_articles(a_vec)
    print(f"   Decoded Articles: {decoded}")
    assert decoded == ["a1", "a2"], "Inverse transform mismatch"

    # 4. Data Loading (Subsampled for Speed)
    print("\n[3/5] Loading and Subsampling Data...")
    # Load data (this reads the generated parquet files)
    # We force reload to ensure we are working with raw data for the demo
    train_df, val_df = load_and_filter_data(load_cached_data=False)

    print(f"   Full Train Shape: {train_df.shape}")
    print(f"   Full Val Shape:   {val_df.shape}")

    # Subsample to speed up model fitting for this demonstration
    # We take the last 100,000 transactions to ensure we have 'recent' data for the time decay
    subset_size = 100000
    train_subset = train_df.tail(subset_size).copy().reset_index(drop=True)

    # Select a few validation customers for prediction
    val_customers = val_df["customer_id"].unique()[:50]
    val_subset = val_df[val_df["customer_id"].isin(val_customers)].copy()

    print(f"   Subset Train Shape: {train_subset.shape}")
    print(f"   Target Customers:   {len(val_customers)}")

    # 5. Model Training
    print("\n[4/5] Training TimeWeightedCooccurrence Model...")
    # Instantiate model
    model = TimeWeightedCooccurrence(decay_rate=2.5, top_k=12)

    # Fit on subset
    # load_cached_data=False ensures we compute the matrix for this specific subset
    # and don't load a potentially huge matrix from a previous run
    model.fit(train_subset, load_cached_data=False)

    print("   Model fitted.")
    print(f"   Global Popularity (Top 5): {model.global_popularity[:5]}")

    # 6. Prediction and Evaluation
    print("\n[5/5] Generating Predictions and Evaluating...")

    # Predict
    # We use train_subset as history. In a real scenario, we might use full history.
    preds_df = model.predict(val_customers, train_subset)

    print("   Predictions generated.")
    print(preds_df.head(3))

    # Validate Output Format
    assert "customer_id" in preds_df.columns
    assert "prediction" in preds_df.columns
    assert len(preds_df) == len(val_customers)

    # Check content format
    first_pred = preds_df.iloc[0]["prediction"]
    assert isinstance(first_pred, str)
    assert len(first_pred.split()) <= 12

    # Evaluate
    map_score = calculate_map12(val_subset, preds_df)
    print(f"   MAP@12 on Subset: {map_score:.6f}")

    # Since we trained on a tiny subset, the score might be low, but it should be calculable.
    # We just assert it returns a float.
    assert isinstance(map_score, float)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
