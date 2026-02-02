import pandas as pd
import numpy as np
import os
from library import config, data_loader, model, inference


def compute_ap_12(actuals, preds):
    """
    Computes Average Precision @ 12 for a single user.
    """
    if not actuals:
        return 0.0

    ap = 0.0
    num_hits = 0.0

    # Truncate predictions to top 12
    preds = preds[:12]

    for i, p in enumerate(preds):
        if p in actuals:
            num_hits += 1
            ap += num_hits / (i + 1)

    return ap / min(len(actuals), 12)


def run_validation():
    """
    Performs validation on the hold-out set (val_metadata.parquet).
    Splits val set into History and Ground Truth (last 7 days).
    Trains model on Train set only.
    Evaluates MAP@12.
    """
    print("=" * 30)
    print("STARTING VALIDATION")
    print("=" * 30)

    # 1. Load Validation Data
    print("Loading Validation Data...")
    val_df = pd.read_parquet(config.VAL_PATH)
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    # 2. Split Validation Data (Last 7 days as target)
    max_date = val_df["t_dat"].max()
    split_date = max_date - pd.Timedelta(days=7)

    print(f"Splitting Validation data at {split_date}...")
    history_df = val_df[val_df["t_dat"] <= split_date].copy()
    ground_truth_df = val_df[val_df["t_dat"] > split_date].copy()

    # Add weights to history_df for the model
    history_df = data_loader.add_time_weights(history_df)

    # 3. Train Model on Training Set ONLY
    print("Loading Training Data...")
    # use_all_data=False ensures we only load train_metadata.parquet
    train_df = data_loader.load_transactions(load_cached_data=True, use_all_data=False)

    # Filter training data to prevent look-ahead bias
    print(f"Filtering Training Data to <= {split_date}...")
    train_df = train_df[train_df["t_dat"] <= split_date].copy()

    print("Fitting Model on Training Data...")
    graph_model = model.TimeAwareTransitionGraph()
    # Force fit (load_cached_data=False) to ensure we don't load a model trained on full data
    graph_model.fit(train_df, load_cached_data=False)

    # 4. Prepare Inputs for Validation Users
    print("Preparing Validation Inputs...")
    # Extract last purchase from the history part of validation
    # We use the helper but must be careful about caching.
    # We pass load_cached_data=False to avoid reading/writing the main cache file with partial data
    val_input_history = data_loader.get_last_purchases(
        history_df, load_cached_data=False
    )

    # Identify users to predict for (those who actually bought something in the test window)
    target_users = ground_truth_df["customer_id"].unique()

    # Create input dataframe for the model
    pred_input = pd.DataFrame({"customer_id": target_users})
    pred_input = pred_input.merge(val_input_history, on="customer_id", how="left")

    # Fill missing history with -1 (Cold Start in Validation Period)
    pred_input["article_id"] = pred_input["article_id"].fillna(-1).astype(np.int64)
    if "weight" in pred_input.columns:
        pred_input["weight"] = pred_input["weight"].fillna(0.0)

    # 5. Generate Predictions
    print(f"Generating predictions for {len(pred_input)} validation users...")
    preds_df = graph_model.generate_predictions(pred_input)

    # 6. Compute MAP@12
    print("Computing MAP@12...")
    # Create dictionary of ground truth: {cust_id: set(article_ids)}
    ground_truth = (
        ground_truth_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    # Create dictionary of predictions: {cust_id: list(article_ids)}
    # Predictions are space-separated strings
    preds_map = dict(zip(preds_df["customer_id"], preds_df["prediction"]))

    ap_scores = []
    users = []

    for uid in target_users:
        actuals = ground_truth.get(uid, set())
        pred_str = preds_map.get(uid, "")

        # Parse string to list of ints
        if pred_str:
            preds = [int(x) for x in pred_str.split()]
        else:
            preds = []

        ap = compute_ap_12(actuals, preds)
        ap_scores.append(ap)
        users.append(uid)

    mean_ap = np.mean(ap_scores)
    print(f"Final Validation Metric: {mean_ap:.10f}")

    return pd.DataFrame({"customer_id": users, "ap": ap_scores})


def failure_analysis(ap_df):
    """
    Correlates model performance (AP) with customer features.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    customers_df = pd.read_csv(config.CUSTOMERS_PATH)

    # Merge AP scores with customer metadata
    analysis_df = ap_df.merge(customers_df, on="customer_id", how="left")

    # Preprocessing for correlation
    # Fill missing Age with median
    analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].median())

    # Factorize categorical columns to numeric codes
    # Handle NaNs by treating them as a separate category (-1)
    analysis_df["fashion_news_frequency"] = pd.factorize(
        analysis_df["fashion_news_frequency"], use_na_sentinel=False
    )[0]
    analysis_df["club_member_status"] = pd.factorize(
        analysis_df["club_member_status"], use_na_sentinel=False
    )[0]

    features = ["age", "fashion_news_frequency", "club_member_status"]

    print("Correlation between Customer Features and Error (AP Score):")
    print("(Positive correlation = Feature associates with BETTER performance)")
    print("(Negative correlation = Feature associates with WORSE performance)")

    for feat in features:
        if analysis_df[feat].nunique() > 1:
            corr = analysis_df[feat].corr(analysis_df["ap"])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: No variance")


def main():
    # 1. Setup
    config.set_seed()

    # 2. Validation & Failure Analysis
    # We run this first to assess the model quality on unseen data
    ap_df = run_validation()
    failure_analysis(ap_df)

    # 3. Final Inference
    val_map = ap_df["ap"].mean()
    if val_map > 0.026059042:
        print("\n" + "=" * 30)
        print("FINAL INFERENCE")
        print("=" * 30)
        # Retrain on ALL data (Train + Val) for the submission
        # We disable cache loading to force a fresh fit on the combined dataset
        inference.run_inference(load_cached_data=False)
    else:
        print(
            f"\nValidation score {val_map:.9f} did not beat threshold 0.026059042. Skipping inference."
        )


if __name__ == "__main__":
    main()
