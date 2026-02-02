import pandas as pd
import numpy as np
import torch
import gc
import random
import sys
from pathlib import Path
from sklearn.metrics import average_precision_score

# Import library modules
from library.config import Config
from library.data_factory import DataFactory
from library.embedder import LatentEmbedder
from library.retrieval import CandidateEngine
from library.features import FeatureEngineer
from library.ranker import LGBMRanker

# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_at_12(predictions, ground_truth):
    """
    Computes MAP@12.
    predictions: DataFrame [customer_id, prediction (space separated string)]
    ground_truth: DataFrame [customer_id, article_id]
    """
    # Group ground truth into list of articles per customer
    gt_grouped = ground_truth.groupby("customer_id")["article_id"].apply(set).to_dict()

    # Parse predictions
    preds_dict = dict(zip(predictions["customer_id"], predictions["prediction"]))

    ap_sum = 0.0
    n_customers = 0

    for cust_id, pred_str in preds_dict.items():
        if cust_id not in gt_grouped:
            continue

        n_customers += 1
        actuals = gt_grouped[cust_id]

        # Get top 12
        preds = pred_str.split()[:12]

        score = 0.0
        num_hits = 0

        for i, p in enumerate(preds):
            if p in actuals:
                num_hits += 1
                score += num_hits / (i + 1.0)

        # Metric definition: divide by min(m, 12)
        denom = min(len(actuals), 12)
        if denom > 0:
            ap_sum += score / denom

    if n_customers == 0:
        return 0.0

    return ap_sum / n_customers


def calculate_user_ap(row, gt_dict):
    """Helper for failure analysis"""
    cust_id = row["customer_id"]
    if cust_id not in gt_dict:
        return np.nan

    actuals = gt_dict[cust_id]
    preds = str(row["prediction"]).split()[:12]

    score = 0.0
    num_hits = 0

    for i, p in enumerate(preds):
        if p in actuals:
            num_hits += 1
            score += num_hits / (i + 1.0)

    denom = min(len(actuals), 12)
    return score / denom if denom > 0 else 0.0


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------


def main():
    print("Starting End-to-End Pipeline...")
    set_seed(Config.SEED)

    # 1. Load Data
    print("\n[1] Loading Data")
    full_df = DataFactory.load_full_data()
    train_history, val_ground_truth = DataFactory.get_time_split(full_df)

    # Load Metadata
    customers = pd.read_csv(Config.INPUT_DIR / "customers.csv")
    articles = pd.read_csv(Config.INPUT_DIR / "articles.csv")

    # Subsample Validation Users for Fast Baseline Training
    # We select users who actually have ground truth in the validation period
    val_users_with_gt = val_ground_truth["customer_id"].unique()

    # Shuffle
    rng = np.random.default_rng(Config.SEED)
    rng.shuffle(val_users_with_gt)

    # Split for Ranker Training (50k) and Ranker Validation (10k)
    # This keeps runtime low while providing enough signal
    n_ranker_train = 50000
    n_ranker_val = 10000

    ranker_train_users = val_users_with_gt[:n_ranker_train]
    ranker_val_users = val_users_with_gt[n_ranker_train : n_ranker_train + n_ranker_val]

    print(f"Ranker Train Users: {len(ranker_train_users)}")
    print(f"Ranker Val Users: {len(ranker_val_users)}")

    # 2. Train Retrieval Models (On History)
    print("\n[2] Training Retrieval Models")
    embedder = LatentEmbedder()
    embedder.fit(train_history)

    engine = CandidateEngine()
    engine.embedder = embedder  # Share the trained embedder
    engine.fit(train_history)  # Builds Co-occurrence and Popularity

    # 3. Generate Candidates & Features for Ranker Training
    print("\n[3] Generating Ranker Training Data")

    feature_engineer = FeatureEngineer()
    feature_engineer.embedder = embedder

    def process_split(user_ids, split_name):
        users_df = pd.DataFrame({"customer_id": user_ids})

        # Generate Candidates
        cands = engine.generate_candidates(
            users_df, train_history, load_cached_data=False
        )

        # Labeling
        # Create a set of (user, article) present in val_ground_truth
        # Optimization: Filter val_ground_truth to relevant users first
        relevant_gt = val_ground_truth[val_ground_truth["customer_id"].isin(user_ids)]
        gt_set = set(zip(relevant_gt["customer_id"], relevant_gt["article_id"]))

        # Apply labels
        # Vectorized check
        cands_set = set(zip(cands["customer_id"], cands["article_id"]))
        # Intersection
        positives = cands_set.intersection(gt_set)

        # Map back to dataframe
        # We can construct a lookup
        pos_lookup = {pair: 1 for pair in positives}

        def get_label(row):
            return pos_lookup.get((row["customer_id"], row["article_id"]), 0)

        # Using a MultiIndex map is faster than apply
        cands["target"] = cands.set_index(["customer_id", "article_id"]).index.map(
            lambda x: pos_lookup.get(x, 0)
        )

        print(f"[{split_name}] Positives: {cands['target'].sum()} / {len(cands)}")

        # Generate Features
        feats = feature_engineer.generate_features(
            cands, train_history, customers, articles, load_cached_data=False
        )
        return feats

    ranker_train_df = process_split(ranker_train_users, "Ranker Train")
    ranker_val_df = process_split(ranker_val_users, "Ranker Val")

    # 4. Train Ranker
    print("\n[4] Training Ranker")
    ranker = LGBMRanker()
    ranker.fit(ranker_train_df, ranker_val_df)

    # 5. Validation & Metrics
    print("\n[5] Evaluating Model")
    # Predict on ranker_val_df
    val_preds = ranker.predict(ranker_val_df, output_path=None, load_model=False)

    # Filter ground truth to ranker_val_users
    val_gt_subset = val_ground_truth[
        val_ground_truth["customer_id"].isin(ranker_val_users)
    ]

    final_map = map_at_12(val_preds, val_gt_subset)
    print(f"Final Validation Metric: {final_map:.16f}")

    # 6. Failure Analysis
    print("\n[6] Failure Analysis")
    # Calculate per-user AP
    gt_dict = val_gt_subset.groupby("customer_id")["article_id"].apply(set).to_dict()
    val_preds["ap"] = val_preds.apply(
        lambda row: calculate_user_ap(row, gt_dict), axis=1
    )

    # Merge with customer metadata
    analysis_df = val_preds.merge(customers, on="customer_id", how="left")

    # Correlations
    # Fill NaNs for correlation
    if "age" in analysis_df.columns:
        analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].mean())
        corr_age = analysis_df["ap"].corr(analysis_df["age"])
        print(f"Correlation (AP vs Age): {corr_age:.4f}")

    # Check correlation with fashion_news_frequency (convert to code if needed)
    if "fashion_news_frequency" in analysis_df.columns:
        # Simple mapping
        fn_map = {"NONE": 0, "Regularly": 1, "Monthly": 0.5}
        analysis_df["fn_code"] = (
            analysis_df["fashion_news_frequency"].map(fn_map).fillna(0)
        )
        corr_fn = analysis_df["ap"].corr(analysis_df["fn_code"])
        print(f"Correlation (AP vs Fashion News Freq): {corr_fn:.4f}")

    # 7. Submission
    threshold = 0.024413818879111926
    if final_map > threshold:
        print(
            f"\n[7] Metric ({final_map:.6f}) > Threshold ({threshold:.6f}). Generating Submission..."
        )

        # A. Inference Retraining (Shift Windows)
        print("   -> Retraining Retrieval Models on Full Data...")
        # Re-fit embedder on full data
        embedder.fit(full_df, load_cached_data=False)
        # Re-fit engine on full data
        engine.fit(full_df)

        # B. Test Prediction
        print("   -> Loading Test Customers...")
        test_customers = DataFactory.load_test_customers()

        # To avoid OOM, we can process test set in chunks, but 220GB is enough for 1.3M users * 60 cands
        # We will process in one go for simplicity given the hardware specs.

        print("   -> Generating Test Candidates...")
        test_cands = engine.generate_candidates(
            test_customers, full_df, load_cached_data=False
        )

        print("   -> Generating Test Features...")
        test_feats = feature_engineer.generate_features(
            test_cands, full_df, customers, articles, load_cached_data=False
        )

        print("   -> Predicting...")
        ranker.predict(
            test_feats,
            output_path=Config.SUBMISSION_DIR / "submission.csv",
            load_model=False,
        )

        print("Submission generated successfully.")
    else:
        print(
            f"\n[7] Metric ({final_map:.6f}) <= Threshold ({threshold:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
