import pandas as pd
import numpy as np
import os
import gc
from library.utils import set_seed, mapk
from library.data_loader import DataLoader
from library.generators import CandidateEngine
from library.feature_engineering import FeatureFactory
from library.ranker import LGBMRankerWrapper, generate_submission


def main():
    # 1. Setup
    set_seed(42)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    print("Initializing pipeline...")

    # 2. Data Loading
    loader = DataLoader(
        input_dir=INPUT_DIR, metadata_dir=METADATA_DIR, cache_dir=WORKING_DIR
    )

    # Load content (needed for feature engineering)
    articles, customers = loader.load_content_data()

    # Load transactions (cached if available)
    transactions = loader.load_transactions(load_cached_data=True)

    # Initialize Engines
    candidate_engine = CandidateEngine(transactions, cache_dir=WORKING_DIR)
    feature_factory = FeatureFactory(
        transactions, articles, customers, cache_dir=WORKING_DIR
    )

    # 3. Training Phase (Week 1)
    print("\n=== Training Phase (Target Week 1) ===")
    TRAIN_WEEK = 1

    # Identify customers active in Week 1
    train_transactions = transactions[transactions["week"] == TRAIN_WEEK]
    train_customers_all = train_transactions["customer_id"].unique()

    # Subsample for speed (max 1k users) since we are using a heuristic ranker that doesn't need training
    N_TRAIN_SAMPLES = 1000
    if len(train_customers_all) > N_TRAIN_SAMPLES:
        np.random.seed(42)
        train_customers = np.random.choice(
            train_customers_all, size=N_TRAIN_SAMPLES, replace=False
        )
        print(
            f"Subsampled training customers from {len(train_customers_all)} to {len(train_customers)}"
        )
    else:
        train_customers = train_customers_all
        print(f"Using all {len(train_customers)} training customers")

    # Generate Candidates
    candidates_train = candidate_engine.generate_candidates(
        train_customers, target_week=TRAIN_WEEK, load_cached_data=True
    )

    # Create Features
    features_train = feature_factory.create_features(
        candidates_train, target_week=TRAIN_WEEK, load_cached_data=True
    )

    # Prepare Ranking Data
    X_train, y_train, group_train = feature_factory.get_ranking_data(features_train)

    # Train Ranker
    # Using moderate n_estimators for speed while maintaining performance
    ranker_params = {
        "n_estimators": 150,
        "learning_rate": 0.1,
        "n_jobs": 12,
        "verbose": -1,
    }
    ranker = LGBMRankerWrapper(params=ranker_params)

    print("Fitting model...")
    ranker.fit(X_train, y_train, group_train)

    # Clean up training data to free memory
    del (
        candidates_train,
        features_train,
        X_train,
        y_train,
        group_train,
        train_transactions,
    )
    gc.collect()

    # 4. Validation Phase (Week 0)
    print("\n=== Validation Phase (Target Week 0) ===")
    VAL_WEEK = 0

    # Identify validation customers (active in Week 0)
    val_transactions = transactions[transactions["week"] == VAL_WEEK]
    val_customers = val_transactions["customer_id"].unique()
    print(f"Validating on {len(val_customers)} customers")

    # Generate Candidates
    candidates_val = candidate_engine.generate_candidates(
        val_customers, target_week=VAL_WEEK, load_cached_data=True
    )

    # Create Features
    features_val = feature_factory.create_features(
        candidates_val, target_week=VAL_WEEK, load_cached_data=True
    )

    # Prepare Data
    X_val, y_val, group_val = feature_factory.get_ranking_data(features_val)

    # Predict
    print("Predicting validation scores...")
    scores_val = ranker.predict(X_val)

    # Attach scores
    features_val["score"] = scores_val

    # 5. Metric Calculation
    print("Calculating MAP@12...")

    # Get Top 12 Predictions
    # Sort by customer and score
    val_preds_df = features_val.sort_values(
        ["customer_id", "score"], ascending=[True, False]
    )
    # Take top 12
    val_preds_df = val_preds_df.groupby("customer_id").head(12)
    # Aggregate to list
    preds_map = val_preds_df.groupby("customer_id")["article_id"].apply(list)

    # Get Ground Truth
    truth_map = val_transactions.groupby("customer_id")["article_id"].apply(list)

    # Align indices
    # We evaluate on all customers present in the validation week
    common_users = truth_map.index

    actual_list = truth_map.tolist()
    # If a user has no predictions (e.g. no candidates generated), they get an empty list
    predicted_list = [preds_map.get(u, []) for u in common_users]

    val_map = mapk(actual_list, predicted_list, k=12)

    print(f"Final Validation Metric: {val_map}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (abs difference between binary label and score)
    features_val["error"] = (features_val["label"] - features_val["score"]).abs()

    # Select numeric features for correlation
    numeric_cols = features_val.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude IDs and target/error/score
    exclude = ["customer_id", "article_id", "label", "error", "score"]
    feature_cols = [c for c in numeric_cols if c not in exclude]

    correlations = features_val[feature_cols + ["error"]].corr()["error"].drop("error")
    correlations = correlations.abs().sort_values(ascending=False)

    print("Top 5 features correlated with error magnitude:")
    print(correlations.head(5))

    # Clean up validation data
    del (
        candidates_val,
        features_val,
        X_val,
        y_val,
        group_val,
        val_transactions,
        val_preds_df,
    )
    gc.collect()

    # 7. Submission Phase
    THRESHOLD = 0.02648135503151091
    if val_map > THRESHOLD:
        print("\n=== Submission Phase ===")

        # Load Test Customers
        test_customers_df = loader.load_test_customers()
        test_customers = test_customers_df["customer_id"].unique()
        print(f"Generating predictions for {len(test_customers)} test customers...")

        TEST_WEEK = -1

        # Generate Candidates (using full history)
        candidates_test = candidate_engine.generate_candidates(
            test_customers, target_week=TEST_WEEK, load_cached_data=True
        )

        # Create Features
        features_test = feature_factory.create_features(
            candidates_test, target_week=TEST_WEEK, load_cached_data=True
        )

        # Prepare Data
        X_test, _, _ = feature_factory.get_ranking_data(features_test)

        # Predict
        scores_test = ranker.predict(X_test)

        # Generate Submission
        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

        generate_submission(features_test, scores_test, sample_sub_path, output_path)

    else:
        print(
            f"Validation metric {val_map} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
