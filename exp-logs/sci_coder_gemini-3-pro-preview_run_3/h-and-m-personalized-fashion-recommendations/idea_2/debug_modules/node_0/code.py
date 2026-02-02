import pandas as pd
import numpy as np
import os
import sys
import library.config as config
import library.data_loader as data_loader
from library.retrieval import SparseGraphRetriever
from library.features import FeatureEngineer
from library.ranker import LGBMRanker

# Set random seeds for reproducibility
np.random.seed(config.SEED)


def run_demo():
    print("Starting H&M Recommendation Pipeline Demo...")

    # =========================================================================
    # 1. Data Loading and Sampling
    # =========================================================================
    print("\n[Step 1] Loading and Sampling Data...")

    # Load the training data split (History vs Target)
    # df_history: Data used to build the graph and features
    # df_target: Data used to generate ground truth labels for the ranker
    df_history_full, df_target_full = data_loader.load_train_data_split(
        val_days=config.VAL_DAYS,
        load_cached_data=False,  # Force reload to demonstrate logic
    )

    # Sample data for speed
    # We take a subset of recent history to ensure we have active users
    SAMPLE_HISTORY_SIZE = 50000
    df_history_sample = df_history_full.tail(SAMPLE_HISTORY_SIZE).copy()

    # Identify users present in the sampled history
    valid_users = df_history_sample[config.USER_COL].unique()

    # Filter targets to only include these users
    df_target_sample = df_target_full[
        df_target_full[config.USER_COL].isin(valid_users)
    ].copy()

    # Further limit the number of users for training the ranker to 500
    train_users = df_target_sample[config.USER_COL].unique()[:500]
    df_target_sample = df_target_sample[
        df_target_sample[config.USER_COL].isin(train_users)
    ]

    print(f"Sampled History: {len(df_history_sample)} rows")
    print(f"Sampled Target: {len(df_target_sample)} rows (Users: {len(train_users)})")

    # =========================================================================
    # 2. Retrieval (Stage 1)
    # =========================================================================
    print("\n[Step 2] Stage 1: Retrieval (Sparse Graph)...")

    # Initialize Retriever
    # We reduce top_k to 20 for this demo to speed up processing
    retriever = SparseGraphRetriever(
        decay_rate=config.DECAY_RATE, history_weight=config.HISTORY_WEIGHT, top_k=20
    )

    # Fit the transition matrix on the sampled history
    retriever.fit(df_history_sample, load_cached_data=False)

    # Verify transition matrix was created
    assert retriever.transition_matrix is not None
    assert retriever.transition_matrix.shape[0] == retriever.n_articles

    # Query candidates for the training users
    print("Generating candidates for training users...")
    candidates_train = retriever.query(df_history_sample, train_users)

    # Basic validation
    assert not candidates_train.empty
    assert config.USER_COL in candidates_train.columns
    assert config.ITEM_COL in candidates_train.columns
    assert "retrieval_score" in candidates_train.columns

    # =========================================================================
    # 3. Feature Engineering
    # =========================================================================
    print("\n[Step 3] Feature Engineering...")

    fe = FeatureEngineer()

    # Generate features for training
    # This merges metadata and calculates dynamic features like 'days_since_last_purchase'
    # It also attaches labels (1/0) based on df_target_sample
    features_train = fe.generate_features(
        candidates_train,
        df_history_sample,
        mode="train",
        df_target=df_target_sample,
        load_cached_data=False,
    )

    # Verify labels exist
    assert "label" in features_train.columns
    assert features_train["label"].nunique() > 0

    # =========================================================================
    # 4. Ranking (Stage 2)
    # =========================================================================
    print("\n[Step 4] Stage 2: Ranking (LightGBM)...")

    # Configure Ranker for speed (override default config)
    fast_params = config.LGBM_PARAMS.copy()
    fast_params.update(
        {
            "n_estimators": 10,  # Reduce trees
            "num_leaves": 16,  # Reduce complexity
            "early_stopping_rounds": 5,
            "verbose": -1,
        }
    )

    ranker = LGBMRanker(params=fast_params)

    # Train the ranker
    # We use the same set for validation just to demonstrate the API
    ranker.train(features_train, df_val=features_train, save_model=True)

    assert ranker.model is not None
    assert config.CACHE_RANKER_MODEL.exists()

    # =========================================================================
    # 5. Inference and Submission
    # =========================================================================
    print("\n[Step 5] Inference and Submission...")

    # Load Test Users (Customers we need to predict for)
    all_test_users = data_loader.load_test_users()

    # Sample 500 test users for the demo
    test_users_sample = all_test_users[:500]
    print(f"Predicting for {len(test_users_sample)} test users...")

    # 1. Retrieve Candidates for Test Users
    # In a real scenario, we would use the FULL history here.
    # For consistency with the fitted retriever in this demo, we use the sampled history.
    candidates_test = retriever.query(df_history_sample, test_users_sample)

    # 2. Generate Features for Test Candidates
    features_test = fe.generate_features(
        candidates_test, df_history_sample, mode="test", load_cached_data=False
    )

    # 3. Predict and Generate Submission
    # This method predicts scores, selects top 12, merges with the full sample submission,
    # fills missing users with fallback, and saves to CSV.
    ranker.predict(features_test, load_cached_model=True)

    # =========================================================================
    # 6. Final Validation
    # =========================================================================
    print("\n[Step 6] Validating Output...")

    if config.SUBMISSION_PATH.exists():
        sub_df = pd.read_csv(config.SUBMISSION_PATH)

        # Check dimensions (should match sample submission row count)
        # Note: data_loader.load_test_users() loads from test_metadata which matches sample_submission
        expected_rows = len(all_test_users)
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(sub_df)}"

        # Check columns
        assert config.USER_COL in sub_df.columns
        assert "prediction" in sub_df.columns

        # Check format of prediction (string)
        first_pred = sub_df.iloc[0]["prediction"]
        assert isinstance(first_pred, str)
        assert len(first_pred.split()) <= 12

        print(f"Submission generated successfully at: {config.SUBMISSION_PATH}")
        print("Demo completed successfully.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    run_demo()
