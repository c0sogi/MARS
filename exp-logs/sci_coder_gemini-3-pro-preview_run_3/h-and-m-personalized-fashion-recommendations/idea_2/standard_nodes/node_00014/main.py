import pandas as pd
import numpy as np
import os
import gc
from datetime import timedelta
from sklearn.metrics import average_precision_score

# Import provided libraries
import library.config as config
import library.data_loader as data_loader
import library.retrieval as retrieval

# Set Seeds for reproducibility
np.random.seed(config.SEED)


def map_at_12(df_preds, df_truth):
    """
    Computes MAP@12 metric.

    Args:
        df_preds (pd.DataFrame): Dataframe with columns [customer_id, prediction].
                                 Prediction is a space-separated string of article_ids.
        df_truth (pd.DataFrame): Dataframe with columns [customer_id, article_id].
                                 Contains the actual purchases (ground truth).

    Returns:
        float: The mean average precision at 12.
        pd.DataFrame: A dataframe containing the AP score for each user.
    """
    # Group truth into a set of article_ids per user for O(1) lookup
    truth_dict = df_truth.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()

    # Convert predictions to a series for iteration
    if config.USER_COL in df_preds.columns:
        preds_series = df_preds.set_index(config.USER_COL)["prediction"]
    else:
        preds_series = df_preds

    scores = []
    users = []

    # Iterate over users in the prediction set
    for user, pred_str in preds_series.items():
        # If user has no ground truth, skip
        if user not in truth_dict:
            continue

        actual = truth_dict[user]
        if not actual:
            continue

        # Parse predictions (top 12)
        predicted_strs = pred_str.split()[:12]

        score = 0.0
        num_hits = 0

        for i, p_str in enumerate(predicted_strs):
            # Convert prediction string to int to match ground truth format (int64)
            # e.g. "0108775015" -> 108775015
            try:
                p_val = int(p_str)
            except ValueError:
                p_val = -1

            if p_val in actual:
                num_hits += 1
                score += num_hits / (i + 1.0)

        # Average Precision for this user
        ap = score / min(len(actual), 12)
        scores.append(ap)
        users.append(user)

    return np.mean(scores), pd.DataFrame({config.USER_COL: users, "ap": scores})


def run():
    print("Starting execution...")

    # =========================================================================
    # 1. PREPARE RANKER TRAINING DATA
    # =========================================================================
    print("\n[Step 1] Preparing Ranker Training Data")

    # Load training data split (History vs Target)
    # We use the last 7 days of the training set as the target for the ranker
    df_hist, df_target = data_loader.load_train_data_split(val_days=7)

    # Optimization: Sample users for ranker training to ensure speed
    # 50,000 users provide enough signal for the ranker while keeping runtime low
    TRAIN_USER_SAMPLE = 50000
    unique_users = df_target[config.USER_COL].unique()

    if len(unique_users) > TRAIN_USER_SAMPLE:
        print(f"Sampling {TRAIN_USER_SAMPLE} users for ranker training...")
        train_users = np.random.choice(unique_users, TRAIN_USER_SAMPLE, replace=False)
    else:
        train_users = unique_users

    # Initialize and Fit Retriever on History
    print("Fitting Retriever on Training History...")
    retriever = retrieval.SparseGraphRetriever()
    retriever.fit(df_hist)

    # Generate Candidates for the sampled training users
    print("Generating Training Candidates...")
    train_candidates = retriever.query(df_hist, train_users)

    # Generate Features (and Labels) for the candidates
    print("Generating Training Features...")
    feat_eng = features.FeatureEngineer()
    df_train_features = feat_eng.generate_features(
        train_candidates, df_hist, mode="train", df_target=df_target, split_name="train"
    )

    # Clean up memory
    del df_hist, df_target, train_candidates
    gc.collect()

    # =========================================================================
    # 2. TRAIN RANKER
    # =========================================================================
    print("\n[Step 2] Training Ranker")

    # Split generated features into Train/Val for LightGBM early stopping
    users_train = df_train_features[config.USER_COL].unique()
    np.random.shuffle(users_train)
    n_split = int(len(users_train) * 0.8)
    u_tr = users_train[:n_split]
    u_val = users_train[n_split:]

    train_set = df_train_features[df_train_features[config.USER_COL].isin(u_tr)]
    val_set = df_train_features[df_train_features[config.USER_COL].isin(u_val)]

    # Initialize and Train Ranker
    lgbm_ranker = ranker.LGBMRanker()
    lgbm_ranker.train(train_set, val_set)

    # Clean up memory
    del df_train_features, train_set, val_set
    gc.collect()

    # =========================================================================
    # 3. VALIDATION (HOLD-OUT SET)
    # =========================================================================
    print("\n[Step 3] Validation on Hold-Out Set")

    # Load Validation Metadata
    df_val_meta = pd.read_parquet(config.VAL_METADATA_PATH)
    df_val_meta[config.DATE_COL] = pd.to_datetime(df_val_meta[config.DATE_COL])

    # Split Validation set into History and Ground Truth (Last 7 days)
    # This simulates the test scenario for the validation users
    max_date = df_val_meta[config.DATE_COL].max()
    split_date = max_date - timedelta(days=7)

    val_history = df_val_meta[df_val_meta[config.DATE_COL] <= split_date].copy()
    val_truth = df_val_meta[df_val_meta[config.DATE_COL] > split_date].copy()
    val_users = val_truth[config.USER_COL].unique()

    print(f"Validation Users: {len(val_users)}")

    # Prepare History for Retrieval: Train History + Validation History
    # We reload train history to combine it
    df_train_hist_full, _ = data_loader.load_train_data_split(val_days=7)
    full_val_history = pd.concat([df_train_hist_full, val_history], ignore_index=True)

    # Refit Retriever on the combined history
    # load_cached_data=False forces a rebuild of the transition matrix
    print("Refitting Retriever for Validation...")
    retriever.fit(full_val_history, load_cached_data=False)

    # Generate Candidates for Validation Users
    print("Generating Validation Candidates...")
    val_candidates = retriever.query(full_val_history, val_users)

    # Generate Features for Validation Candidates
    print("Generating Validation Features...")
    val_features = feat_eng.generate_features(
        val_candidates,
        full_val_history,
        mode="test",
        split_name="val",  # No labels needed for inference
    )

    # Predict using Ranker
    print("Predicting on Validation Set...")
    # We access the model directly to get raw scores for metric calculation
    # Ensure feature columns match training
    feature_cols = lgbm_ranker.feature_cols
    if feature_cols is None:
        feature_cols = lgbm_ranker._get_feature_columns(val_features)

    scores = lgbm_ranker.model.predict(val_features[feature_cols])
    val_features["score"] = scores

    # Select Top 12 Recommendations
    print("Selecting Top 12...")
    top_k = val_features.sort_values(
        [config.USER_COL, "score"], ascending=[True, False]
    )
    top_k = top_k.groupby(config.USER_COL).head(12)

    # Format predictions for MAP calculation (space-separated strings)
    top_k["article_id_str"] = top_k[config.ITEM_COL].astype(str).str.zfill(10)
    preds_df = (
        top_k.groupby(config.USER_COL)["article_id_str"]
        .apply(" ".join)
        .reset_index(name="prediction")
    )

    # Calculate MAP@12
    metric, user_ap_df = map_at_12(preds_df, val_truth)
    print(f"Final Validation Metric: {metric:.10f}")

    # Clean up memory
    del df_train_hist_full, full_val_history, val_history, val_candidates
    gc.collect()

    # =========================================================================
    # 4. FAILURE ANALYSIS
    # =========================================================================
    print("\n[Step 4] Failure Analysis")

    # Merge AP scores with user metadata to find correlations
    # We extract static user features from the validation feature matrix
    user_meta_cols = [config.USER_COL, "age", "FN", "Active", "club_member_status_idx"]
    # Ensure columns exist
    available_cols = [c for c in user_meta_cols if c in val_features.columns]

    user_meta = val_features.drop_duplicates(subset=[config.USER_COL])[available_cols]
    analysis_df = user_ap_df.merge(user_meta, on=config.USER_COL, how="left")

    # Calculate correlations
    # Select only numeric columns for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    if "ap" in numeric_cols:
        correlations = analysis_df[numeric_cols].corr()["ap"]
        print("Correlation of Error (AP) with Features:")
        print(correlations.drop("ap"))

    # =========================================================================
    # 5. SUBMISSION
    # =========================================================================
    THRESHOLD = 0.025945579

    if metric > THRESHOLD:
        print(
            f"\n[Step 5] Metric ({metric:.5f}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Users
        test_users = data_loader.load_test_users()

        # Load Full Data (Train + Val) for final training history
        print("Loading full history for inference...")
        df_train_full = pd.read_parquet(config.TRAIN_METADATA_PATH)
        df_val_full = pd.read_parquet(config.VAL_METADATA_PATH)
        full_data = pd.concat([df_train_full, df_val_full], ignore_index=True)
        full_data[config.DATE_COL] = pd.to_datetime(full_data[config.DATE_COL])

        # Refit Retriever on Full Data
        print("Refitting Retriever on Full Data...")
        retriever.fit(full_data, load_cached_data=False)

        # Generate Candidates for Test Users
        print("Generating Test Candidates...")
        test_candidates = retriever.query(full_data, test_users)

        # Generate Features for Test Candidates
        print("Generating Test Features...")
        test_features = feat_eng.generate_features(
            test_candidates, full_data, mode="test", split_name="test"
        )

        # Predict and Save Submission
        # The ranker.predict method handles the top-12 logic and CSV writing
        print("Predicting and Saving Submission...")
        lgbm_ranker.predict(test_features)

    else:
        print(
            f"\n[Step 5] Metric ({metric:.5f}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    run()
