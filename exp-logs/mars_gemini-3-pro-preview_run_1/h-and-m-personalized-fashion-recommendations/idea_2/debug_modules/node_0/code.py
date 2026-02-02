import os
import pandas as pd
import numpy as np
import warnings
from library.utils import set_seed, mapk
from library.data_loader import DataLoader
from library.generators import CandidateEngine
from library.feature_engineering import FeatureFactory
from library.ranker import LGBMRankerWrapper, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Recommendation System Pipeline Demo...")

    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Setup
    os.makedirs(WORKING_DIR, exist_ok=True)
    set_seed(42)

    # ---------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------
    print("\n[1/6] Loading Data...")
    loader = DataLoader(
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
    )

    # Load content data
    articles_df, customers_df = loader.load_content_data()

    # Load transactions (this handles the train/val split merge and week calculation)
    # forcing load_cached_data=False to demonstrate full processing logic once,
    # though the class handles caching internally.
    transactions_df = loader.load_transactions(load_cached_data=False)

    # Verify Data
    print(f"Transactions shape: {transactions_df.shape}")
    print(
        f"Week range: {transactions_df['week'].min()} to {transactions_df['week'].max()}"
    )

    # ---------------------------------------------------------
    # 2. Prepare Training Data (Target Week = 1)
    # ---------------------------------------------------------
    print("\n[2/6] Preparing Training Data (Week 1)...")

    # Identify customers who made purchases in Week 1
    # We sample 2000 customers to keep the demo fast
    train_week_df = transactions_df[transactions_df["week"] == 1]
    train_customers = train_week_df["customer_id"].unique()

    if len(train_customers) > 2000:
        train_customers = np.random.choice(train_customers, 2000, replace=False)

    print(f"Selected {len(train_customers)} customers for training.")

    # Initialize Engines
    cand_engine = CandidateEngine(
        transactions_df, cache_dir=os.path.join(WORKING_DIR, "cache")
    )
    feat_factory = FeatureFactory(
        transactions_df,
        articles_df,
        customers_df,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
    )

    # Generate Candidates for Week 1 (using history > 1)
    # Note: We disable loading from cache to ensure the demo runs the logic
    train_candidates = cand_engine.generate_candidates(
        train_customers, target_week=1, load_cached_data=False
    )

    # Generate Features for Week 1
    train_features = feat_factory.create_features(
        train_candidates, target_week=1, load_cached_data=False
    )

    # ---------------------------------------------------------
    # 3. Prepare Validation Data (Target Week = 0)
    # ---------------------------------------------------------
    print("\n[3/6] Preparing Validation Data (Week 0)...")

    # Identify customers who made purchases in Week 0
    val_week_df = transactions_df[transactions_df["week"] == 0]
    val_customers = val_week_df["customer_id"].unique()

    if len(val_customers) > 1000:
        val_customers = np.random.choice(val_customers, 1000, replace=False)

    print(f"Selected {len(val_customers)} customers for validation.")

    # Generate Candidates for Week 0 (using history > 0)
    val_candidates = cand_engine.generate_candidates(
        val_customers, target_week=0, load_cached_data=False
    )

    # Generate Features for Week 0
    val_features = feat_factory.create_features(
        val_candidates, target_week=0, load_cached_data=False
    )

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[4/6] Training LightGBM Ranker...")

    # Prepare data for LightGBM
    X_train, y_train, group_train = feat_factory.get_ranking_data(train_features)
    X_val, y_val, group_val = feat_factory.get_ranking_data(val_features)

    print(f"Training Features Shape: {X_train.shape}")

    # Initialize Ranker with reduced estimators for speed
    ranker = LGBMRankerWrapper(
        params={
            "n_estimators": 50,
            "learning_rate": 0.1,
            "num_leaves": 20,
            "verbose": -1,
        }
    )

    # Fit Model
    ranker.fit(X_train, y_train, group_train, X_val, y_val, group_val)

    # Show Feature Importance
    importance = ranker.get_feature_importance()
    print("\nTop 5 Features:")
    print(importance.head(5))

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    print("\n[5/6] Evaluating on Validation Set...")

    # Predict scores
    val_scores = ranker.predict(X_val)

    # Assign scores back to dataframe
    val_features["score"] = val_scores

    # Select top 12 predictions per customer
    preds = val_features.sort_values(["customer_id", "score"], ascending=[True, False])
    top_preds = preds.groupby("customer_id").head(12)

    # Prepare Ground Truth for MAP calculation
    # We need the actual purchases from Week 0 for the selected validation customers
    ground_truth_df = val_week_df[val_week_df["customer_id"].isin(val_customers)]
    ground_truth = (
        ground_truth_df.groupby("customer_id")["article_id"].apply(list).to_dict()
    )

    # Prepare Predictions
    predictions = top_preds.groupby("customer_id")["article_id"].apply(list).to_dict()

    # Align lists
    actuals_list = []
    preds_list = []

    # Only evaluate customers who exist in both (which should be all, given our selection logic)
    common_users = set(ground_truth.keys()).intersection(set(predictions.keys()))

    for uid in common_users:
        actuals_list.append(ground_truth[uid])
        preds_list.append(predictions[uid])

    # Calculate MAP@12
    map_score = mapk(actuals_list, preds_list, k=12)
    print(f"Validation MAP@12: {map_score:.5f}")

    # Basic Assertion to ensure model learned something better than random/zero
    # (Note: MAP can be low on small samples, but typically > 0.001)
    if map_score <= 0.0:
        print(
            "Warning: MAP score is 0.0. This might happen with very small samples or difficult splits."
        )

    # ---------------------------------------------------------
    # 6. Inference (Test Set)
    # ---------------------------------------------------------
    print("\n[6/6] Generating Submission for Test Set...")

    # Load Test Customers
    # Sample 1000 for demo purposes
    test_customers_df = loader.load_test_customers()
    test_customers = test_customers_df["customer_id"].values

    if len(test_customers) > 1000:
        test_customers = np.random.choice(test_customers, 1000, replace=False)

    print(f"Selected {len(test_customers)} test customers for inference.")

    # Generate Candidates for Test (Target Week = -1, uses all history)
    test_candidates = cand_engine.generate_candidates(
        test_customers, target_week=-1, load_cached_data=False
    )

    # Generate Features for Test
    test_features = feat_factory.create_features(
        test_candidates, target_week=-1, load_cached_data=False
    )

    # Prepare for Prediction
    X_test, _, _ = feat_factory.get_ranking_data(test_features)

    # Predict
    test_scores = ranker.predict(X_test)

    # Generate Submission File
    submission_path = os.path.join(WORKING_DIR, "submission_demo.csv")
    generate_submission(
        test_features, test_scores, SAMPLE_SUBMISSION_PATH, submission_path
    )

    # Verify Output
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission file generated successfully with shape: {sub_df.shape}")

        # Check format
        example_pred = sub_df.iloc[0]["prediction"]
        print(f"Example Prediction: {example_pred}")
        assert isinstance(example_pred, str), "Prediction should be a string"
        assert len(example_pred.split()) <= 12, "Should not predict more than 12 items"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    main()
