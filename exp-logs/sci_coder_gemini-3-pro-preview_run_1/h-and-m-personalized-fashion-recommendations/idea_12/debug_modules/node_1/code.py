import os
import sys
import pandas as pd
import numpy as np
import shutil
import scipy.sparse as sp

# Import provided library modules
from library import config, data_factory, graph_engine, stratified_inference, evaluation


def setup_demo_environment():
    """
    Sets up a temporary working directory and overrides configuration
    to ensure the demo runs quickly on a small subset of data.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Create a small sample of the training data
    print(">>> Creating sampled dataset from metadata...")
    original_train_path = "./metadata/train.csv"
    sample_train_path = os.path.join(demo_working_dir, "train_sample.csv")

    # FIX: Use entity-based sampling to preserve temporal context (Cite debug_lesson_3)
    # Naive slicing (nrows=50000) resulted in a dataset spanning only 1 day, causing
    # the time-based split to put all data into validation, leaving an empty train set.
    print("Loading full metadata for entity sampling...")
    df_full = pd.read_csv(original_train_path, dtype=config.DTYPE_OPTS)

    unique_customers = df_full["customer_id"].unique()
    # Sample 2000 customers to get a representative dataset size (approx 20k-50k rows)
    # This ensures we have the full date range (2018-2020) for these users.
    sample_size = min(len(unique_customers), 2000)
    # Ensure reproducibility for the sample
    np.random.seed(config.RANDOM_SEED)
    sampled_users = np.random.choice(unique_customers, size=sample_size, replace=False)

    df_sample = df_full[df_full["customer_id"].isin(sampled_users)].copy()
    print(f"Created sample with {len(df_sample)} rows from {sample_size} users.")

    df_sample.to_csv(sample_train_path, index=False)

    # Monkey-patch the config module to use our demo settings
    config.WORKING_DIR = demo_working_dir
    config.TRAIN_META_PATH = sample_train_path

    # Update cache paths in config to point to the new working dir
    config.CACHE_INTERACTION_MATRIX = os.path.join(
        demo_working_dir, "interaction_matrix.npz"
    )
    config.CACHE_SIMILARITY_MATRIX = os.path.join(
        demo_working_dir, "similarity_matrix.npz"
    )
    config.CACHE_USER_HISTORY = os.path.join(demo_working_dir, "user_history.parquet")
    config.CACHE_GLOBAL_TRENDS = os.path.join(demo_working_dir, "global_trend.npy")
    config.CACHE_ITEM_MAP = os.path.join(demo_working_dir, "item_map.parquet")
    config.CACHE_USER_MAP = os.path.join(demo_working_dir, "user_map.parquet")

    # Reduce computational parameters for speed
    config.TRAIN_HISTORY_WEEKS = 10  # Reduce history window
    config.MAX_NEIGHBORS = 10  # Fewer neighbors for similarity
    config.BATCH_SIZE = 1000  # Smaller inference batch

    # Set seeds
    np.random.seed(config.RANDOM_SEED)

    print(">>> Demo environment ready.")


def demo_data_factory():
    print("\n" + "=" * 40)
    print("DEMO: library.data_factory")
    print("=" * 40)

    # 1. Load and Preprocess
    print("Testing load_and_preprocess...")
    df = data_factory.load_and_preprocess(
        config.TRAIN_META_PATH, load_cached_data=False
    )

    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert "days_elapsed" in df.columns, "days_elapsed column missing"
    assert df["article_id"].dtype == "int32", "article_id dtype mismatch"

    print(f"Loaded {len(df)} rows.")

    # 2. Time Split
    print("Testing get_time_split...")
    # Using a small val_days to ensure we have data in both sets given the small sample
    train_df, val_df = data_factory.get_time_split(
        df, val_days=7, train_weeks=config.TRAIN_HISTORY_WEEKS
    )

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}")

    # Validation
    if len(val_df) > 0 and len(train_df) > 0:
        assert (
            train_df["t_dat"].max() <= val_df["t_dat"].min()
        ), "Train data leaked into Validation period"

    return train_df, val_df


def demo_graph_engine(train_df):
    print("\n" + "=" * 40)
    print("DEMO: library.graph_engine")
    print("=" * 40)

    # 1. Mappings
    print("Testing get_mappings...")
    user_map, item_map = graph_engine.get_mappings(train_df, load_cached_data=False)

    n_users = len(user_map)
    n_items = len(item_map)
    print(f"Mapped {n_users} users and {n_items} items.")

    assert n_users > 0 and n_items > 0, "Mappings should not be empty"

    # 2. Interaction Matrix
    print("Testing build_decayed_interaction_matrix...")
    interaction_matrix = graph_engine.build_decayed_interaction_matrix(
        train_df, user_map, item_map, load_cached_data=False
    )

    assert interaction_matrix.shape == (
        n_users,
        n_items,
    ), "Interaction matrix shape mismatch"
    assert sp.issparse(interaction_matrix), "Interaction matrix should be sparse"

    # 3. Similarity Matrix
    print("Testing compute_similarity_matrix...")
    similarity_matrix = graph_engine.compute_similarity_matrix(
        interaction_matrix, load_cached_data=False
    )

    assert similarity_matrix.shape == (
        n_items,
        n_items,
    ), "Similarity matrix shape mismatch"
    # Check diagonal is zero
    diag_sum = np.abs(similarity_matrix.diagonal()).sum()
    assert diag_sum == 0, "Similarity matrix diagonal should be zero"

    return user_map, item_map, similarity_matrix


def demo_stratified_inference(train_df, val_df, user_map, item_map, similarity_matrix):
    print("\n" + "=" * 40)
    print("DEMO: library.stratified_inference")
    print("=" * 40)

    # 1. Instantiate Recommender
    print("Initializing TGSCRecommender...")
    recommender = stratified_inference.TGSCRecommender(
        user_map, item_map, similarity_matrix
    )

    # 2. Fit
    print("Fitting recommender...")
    recommender.fit(train_df, load_cached_data=False)

    assert recommender.global_trends is not None, "Global trends not computed"
    assert recommender.history_matrix is not None, "History matrix not built"

    # 3. Predict
    # Predict for users in the validation set
    val_users = val_df["customer_id"].unique()
    print(f"Predicting for {len(val_users)} validation users...")

    preds_df = recommender.predict(val_users)

    assert isinstance(preds_df, pd.DataFrame), "Prediction output should be a DataFrame"
    assert "customer_id" in preds_df.columns and "prediction" in preds_df.columns
    assert len(preds_df) == len(val_users), "Should have one prediction row per user"

    # Check format of first prediction
    first_pred = preds_df.iloc[0]["prediction"]
    assert isinstance(first_pred, str), "Prediction should be a string"
    pred_items = first_pred.split()
    assert len(pred_items) <= config.TOP_K_PREDICTIONS, "Too many predictions per user"

    return preds_df


def demo_evaluation(preds_df, val_df):
    print("\n" + "=" * 40)
    print("DEMO: library.evaluation")
    print("=" * 40)

    print("Calculating MAP@12...")
    score = evaluation.calculate_map12(preds_df, val_df)

    print(f"Calculated MAP@12: {score:.6f}")
    assert isinstance(score, float), "Score should be a float"
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"


def demo_full_validation_pipeline():
    print("\n" + "=" * 40)
    print("DEMO: Full Validation Pipeline")
    print("=" * 40)

    # Using the validate function which wraps everything
    # We use a debug_sample_size to ensure it runs fast even if it reloads data
    print("Running evaluation.validate()...")
    score = evaluation.validate(load_cached_data=False, debug_sample_size=1000)

    print(f"Pipeline Validation Score: {score:.6f}")
    assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Data Factory
    train_df, val_df = demo_data_factory()

    # 3. Graph Engine
    user_map, item_map, similarity_matrix = demo_graph_engine(train_df)

    # 4. Stratified Inference
    preds_df = demo_stratified_inference(
        train_df, val_df, user_map, item_map, similarity_matrix
    )

    # 5. Evaluation
    demo_evaluation(preds_df, val_df)

    # 6. Full Pipeline Check
    # Note: This will reload data from the config path (which we pointed to our sample)
    demo_full_validation_pipeline()

    print("\n>>> All demonstrations completed successfully.")
