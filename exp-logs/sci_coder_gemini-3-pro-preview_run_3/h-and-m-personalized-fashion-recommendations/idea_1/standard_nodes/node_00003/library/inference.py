import pandas as pd
import numpy as np
from library import config, data_loader, model


def run_inference(load_cached_data=True):
    """
    Orchestrates the inference pipeline: loads data, fits the model,
    generates predictions for the test set, and saves the submission.

    Args:
        load_cached_data (bool): If True, attempts to use cached intermediate files.
    """
    # 1. Set Random Seed
    config.set_seed()
    print("Starting inference pipeline...")

    # 2. Load Transactions
    # We use all available data (train + val) to capture the most recent trends
    # and user history for the test period.
    transactions_df = data_loader.load_transactions(
        load_cached_data=load_cached_data, use_all_data=True
    )

    # 3. Fit Model
    # Initialize and train the graph-based model
    graph_model = model.TimeAwareTransitionGraph()
    graph_model.fit(transactions_df, load_cached_data=load_cached_data)

    # 4. Prepare Test Data
    print("Preparing test customers...")
    # Load the list of customers we need to predict for
    test_customers_df = pd.read_parquet(config.TEST_PATH)

    # Get the last purchase history for all users (for Markov source)
    user_last_purchase_df = data_loader.get_last_purchases(
        transactions_df, load_cached_data=load_cached_data
    )

    # Get top items from history (for Repurchase candidates)
    user_top_items_df = data_loader.get_user_top_items(
        transactions_df, load_cached_data=load_cached_data
    )

    # Merge test customers with their history.
    input_df = test_customers_df[["customer_id"]].merge(
        user_last_purchase_df, on="customer_id", how="left"
    )

    # Merge with top items history
    input_df = input_df.merge(user_top_items_df, on="customer_id", how="left")

    # Fill NaN article_ids with a placeholder (e.g., -1)
    input_df["article_id"] = input_df["article_id"].fillna(-1).astype(np.int64)

    # Note: 'history_items' will be NaN for users with no history.
    # The model handles this check.

    # 5. Generate Predictions
    print(f"Generating predictions for {len(input_df)} customers...")
    predictions_df = graph_model.generate_predictions(input_df)

    # 6. Save Submission
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    # Ensure the output directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    predictions_df.to_csv(config.SUBMISSION_PATH, index=False)
    print("Inference complete.")
