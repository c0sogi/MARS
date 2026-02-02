import pandas as pd
import numpy as np
import sys
import shutil
from pathlib import Path

# Ensure the library can be imported
sys.path.append(str(Path.cwd()))

from library import config, data_loader, model


def run_demonstration():
    print("Initializing Demonstration...")

    # 1. Configuration Setup
    # Set seed for reproducibility
    config.set_seed(42)

    # Override working directory for this demo to avoid cache conflicts
    # and ensure we are testing the 'scratch' generation logic.
    demo_working_dir = Path("./working/demo_execution")
    if demo_working_dir.exists():
        shutil.rmtree(demo_working_dir)
    demo_working_dir.mkdir(parents=True, exist_ok=True)
    config.WORKING_DIR = demo_working_dir

    print(f"Working directory set to: {config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n[Step 1] Loading Transactions...")
    # We load transactions without using cache to demonstrate the processing logic.
    # We use only training data to keep it lighter, though the logic supports validation data too.
    df_transactions = data_loader.load_transactions(
        load_cached_data=False, use_all_data=False
    )

    # Validation
    assert not df_transactions.empty, "Transaction DataFrame should not be empty."
    assert (
        "weight" in df_transactions.columns
    ), "Time-decay weights were not calculated."
    assert "t_dat" in df_transactions.columns, "Date column missing."

    print(f"Loaded {len(df_transactions)} transactions.")

    # OPTIMIZATION: Subsample data for the rest of the demo to ensure speed
    # We take the top 100,000 transactions.
    subset_size = 100000
    if len(df_transactions) > subset_size:
        print(f"Subsampling to top {subset_size} rows for speed...")
        df_subset = df_transactions.head(subset_size).copy()
    else:
        df_subset = df_transactions.copy()

    # 3. User History Extraction
    print("\n[Step 2] Extracting User History...")
    # This function identifies the last item purchased by each customer
    df_history = data_loader.get_last_purchases(df_subset, load_cached_data=False)

    # Validation
    assert "customer_id" in df_history.columns
    assert "article_id" in df_history.columns
    assert df_history[
        "customer_id"
    ].is_unique, "User history should have unique customer_ids."

    print(f"Extracted history for {len(df_history)} unique customers.")

    # 4. Model Training
    print("\n[Step 3] Training Time-Aware Transition Graph...")
    graph_model = model.TimeAwareTransitionGraph()

    # Fit the model on the subset
    graph_model.fit(df_subset, load_cached_data=False)

    # Validation of Model Internals
    assert (
        graph_model.transition_matrix is not None
    ), "Transition matrix was not created."
    assert len(graph_model.global_popularity) > 0, "Global popularity list is empty."

    # Check dimensions
    n_articles = len(graph_model.idx_to_article)
    assert graph_model.transition_matrix.shape == (
        n_articles,
        n_articles,
    ), f"Matrix shape mismatch. Expected ({n_articles}, {n_articles}), got {graph_model.transition_matrix.shape}"

    print(f"Model trained. Vocabulary size: {n_articles} articles.")
    print(
        f"Transition Matrix density: {graph_model.transition_matrix.nnz} stored elements."
    )

    # 5. Prediction Generation
    print("\n[Step 4] Generating Predictions...")

    # Prepare a test input dataframe
    # Case A: Existing users from the history
    test_users_existing = df_history.head(5).copy()

    # Case B: New user (Cold Start) - simulate by creating a user with NaN article_id
    # In the inference pipeline, merge operations result in NaN for users with no history.
    test_users_new = pd.DataFrame(
        {
            "customer_id": ["new_user_001", "new_user_002"],
            "article_id": [np.nan, np.nan],
        }
    )

    # Combine
    input_df = pd.concat([test_users_existing, test_users_new], ignore_index=True)

    # Pre-processing: Fill NaNs with -1 as expected by the model/inference pipeline logic
    # The model's get_indexer will return -1 for these, triggering the fallback to global popularity.
    input_df["article_id"] = input_df["article_id"].fillna(-1).astype(np.int64)

    # Generate
    predictions_df = graph_model.generate_predictions(input_df)

    # Validation of Output
    assert len(predictions_df) == len(input_df), "Output size mismatch."
    assert "customer_id" in predictions_df.columns
    assert "prediction" in predictions_df.columns

    # Validate Prediction Format
    sample_pred = predictions_df.iloc[0]["prediction"]
    assert isinstance(sample_pred, str), "Prediction must be a string."

    pred_items = sample_pred.split()
    assert (
        len(pred_items) == config.TOP_K
    ), f"Prediction length mismatch. Expected {config.TOP_K}, got {len(pred_items)}."

    # Validate Cold Start Fallback
    # The new user should have predictions purely from global popularity (or valid fallbacks)
    cold_start_pred = predictions_df[predictions_df["customer_id"] == "new_user_001"][
        "prediction"
    ].values[0]
    assert len(cold_start_pred.split()) == config.TOP_K

    print("\nSample Output:")
    print(predictions_df.to_string(index=False))

    print("\n[Success] All demonstration steps completed and validated.")


if __name__ == "__main__":
    run_demonstration()
