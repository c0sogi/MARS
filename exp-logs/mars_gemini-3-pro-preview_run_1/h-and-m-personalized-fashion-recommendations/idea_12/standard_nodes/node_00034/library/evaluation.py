import numpy as np
import pandas as pd
import os
from library import config, data_factory, graph_engine, stratified_inference


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k (AP@k) for a single user.

    Args:
        actual (list): List of ground truth items (article_ids).
        predicted (list): List of predicted items (article_ids).
        k (int): Maximum number of predictions to evaluate.

    Returns:
        float: The AP@k score.
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Convert actual to set for O(1) lookup
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        # Check if item is relevant and not a duplicate in predictions
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # Normalize by min(len(actual), k)
    return score / min(len(actual), k)


def calculate_map12(predictions, ground_truth_df):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12).

    Args:
        predictions (dict or pd.DataFrame):
            If dict: {customer_id: [article_id, ...]}
            If DataFrame: Columns ['customer_id', 'prediction'] where prediction is a space-separated string.
        ground_truth_df (pd.DataFrame): DataFrame containing 'customer_id' and 'article_id'.

    Returns:
        float: The MAP@12 score.
    """
    # 1. Prepare Ground Truth
    # Group by customer_id to get list of purchased articles
    # We assume ground_truth_df has 'customer_id' and 'article_id'
    print("Preparing ground truth for evaluation...")
    gt_grouped = (
        ground_truth_df.groupby("customer_id")["article_id"].apply(list).to_dict()
    )

    # 2. Prepare Predictions
    preds_dict = {}
    if isinstance(predictions, pd.DataFrame):
        print("Parsing prediction DataFrame...")
        # Convert DataFrame to dict for faster lookup
        # Prediction column is space-separated string
        # We assume the dataframe has 'customer_id' and 'prediction'

        # Iterating with zip is faster than iterrows
        cust_ids = predictions["customer_id"].values
        pred_strs = predictions["prediction"].values

        for cid, pred_str in zip(cust_ids, pred_strs):
            if pd.isna(pred_str) or pred_str == "":
                preds_dict[cid] = []
            else:
                # Convert space-separated string to list of ints
                # Note: article_ids are int32 in the system, so we convert back to int
                try:
                    preds_dict[cid] = [int(x) for x in str(pred_str).split()]
                except ValueError:
                    # Fallback if IDs are not integers (though config says they are int32)
                    preds_dict[cid] = str(pred_str).split()
    else:
        preds_dict = predictions

    # 3. Calculate MAP
    print("Computing MAP@12...")
    scores = []
    # We only evaluate on customers present in the ground truth (as per metric definition)
    for cid, actual_items in gt_grouped.items():
        pred_items = preds_dict.get(cid, [])
        score = apk(actual_items, pred_items, k=12)
        scores.append(score)

    final_map = np.mean(scores)
    return final_map


def validate(load_cached_data=False, debug_sample_size=None):
    """
    Runs the full validation pipeline using a time-based split.

    Args:
        load_cached_data (bool): Whether to load the raw data from cache.
                                 Note: Graph artifacts are forced to recompute for the split.
        debug_sample_size (int, optional): Number of users to sample for quick debugging.

    Returns:
        float: The validation MAP@12 score.
    """
    print("=" * 40)
    print("STARTING VALIDATION PIPELINE")
    print("=" * 40)

    # 1. Load Data
    # We use the TRAIN_META_PATH which contains the 80% user split.
    # We will further split this by time for validation.
    df = data_factory.load_and_preprocess(
        config.TRAIN_META_PATH, load_cached_data=load_cached_data
    )

    # Debug Sampling
    if debug_sample_size:
        print(f"DEBUG: Sampling {debug_sample_size} users...")
        unique_users = df["customer_id"].unique()
        if len(unique_users) > debug_sample_size:
            np.random.seed(config.RANDOM_SEED)
            sampled_users = np.random.choice(
                unique_users, debug_sample_size, replace=False
            )
            df = df[df["customer_id"].isin(sampled_users)].copy()
            print(f"DEBUG: Data shape after sampling: {df.shape}")

    # 2. Time Split
    # Split last 7 days for validation
    # train_df will have adjusted days_elapsed, val_df will be the ground truth
    train_df, val_df = data_factory.get_time_split(df, val_days=7)

    if len(val_df) == 0:
        raise ValueError(
            "Validation set is empty. Check dataset date range or split parameters."
        )

    # 3. Build Graph Artifacts (Force Recompute for Split)
    # We pass load_cached_data=False to ensure we don't load full-data artifacts
    # This prevents data leakage.
    print("\n[Step 1/3] Building Graph Artifacts for Validation Split...")
    user_map, item_map = graph_engine.get_mappings(train_df, load_cached_data=False)

    interaction_matrix = graph_engine.build_decayed_interaction_matrix(
        train_df, user_map, item_map, load_cached_data=False
    )

    similarity_matrix = graph_engine.compute_similarity_matrix(
        interaction_matrix, load_cached_data=False
    )

    # 4. Fit Recommender
    print("\n[Step 2/3] Fitting TGSC Recommender...")
    recommender = stratified_inference.TGSCRecommender(
        user_map, item_map, similarity_matrix
    )
    # Fit computes global trends and history index on the train_df
    recommender.fit(train_df, load_cached_data=False)

    # 5. Predict
    print("\n[Step 3/3] Generating Predictions...")
    val_customers = val_df["customer_id"].unique()
    preds_df = recommender.predict(val_customers)

    # 6. Score
    print("\n[Result] Calculating Metrics...")
    score = calculate_map12(preds_df, val_df)

    print(f"Validation MAP@12: {score}")
    print("=" * 40)

    return score
