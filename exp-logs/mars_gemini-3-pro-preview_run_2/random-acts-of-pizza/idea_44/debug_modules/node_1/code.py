import sys
import os
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import load_dataset
from library.embedding_generator import generate_embeddings
from library.feature_pipeline import FoldPipeline
from library.model_engine import EnsembleModel


def main():
    print("=== Starting Demo Execution ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Use a very small subset to ensure BERT embeddings and training are instant
    Config.DEBUG_SAMPLE_SIZE = 20

    # Reduce ensemble size and grid search space to minimize runtime
    Config.BAGGING_N_ESTIMATORS = 2
    Config.PARAM_GRID = {
        "C": [1.0],
        "class_weight": [None],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [100],
    }

    # Reduce PCA components to fit within the small sample size (n_components < n_samples)
    Config.AUX_TITLE_PCA_COMPONENTS = 10
    Config.AUX_BODY_PCA_COMPONENTS = 10

    # Set output directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    if not os.path.exists(Config.WORKING_DIR):
        os.makedirs(Config.WORKING_DIR)

    # Set Seed
    set_seed(Config.SEED)
    print("    Configuration applied.")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Dataset...")

    # Load data (force reload to skip cache and verify logic)
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # Verify Data Loading
    print(f"    Train Shape: {df_train.shape}")
    print(f"    Val Shape:   {df_val.shape}")
    print(f"    Test Shape:  {df_test.shape}")

    assert len(df_train) == Config.DEBUG_SAMPLE_SIZE, "Train set size mismatch"
    assert len(df_val) == Config.DEBUG_SAMPLE_SIZE, "Val set size mismatch"
    assert len(df_test) == Config.DEBUG_SAMPLE_SIZE, "Test set size mismatch"

    # Verify critical columns exist
    assert "request_id" in df_train.columns
    assert "requester_received_pizza" in df_train.columns
    assert "requester_received_pizza" not in df_test.columns
    print("    Data loaded and verified.")

    # -------------------------------------------------------------------------
    # 3. Embedding Generation
    # -------------------------------------------------------------------------
    print("\n[3] Generating Embeddings (MiniLM & MPNet)...")

    # Generate embeddings (force recompute)
    embeddings = generate_embeddings(df_train, df_val, df_test, load_cached_data=False)

    # Verify Embedding Structure and Shapes
    expected_splits = ["train", "val", "test"]
    expected_keys = ["anchor_title", "anchor_body", "aux_title", "aux_body"]

    for split in expected_splits:
        assert split in embeddings, f"Missing split {split} in embeddings"
        for key in expected_keys:
            assert key in embeddings[split], f"Missing key {key} in {split} embeddings"

            emb_matrix = embeddings[split][key]
            assert len(emb_matrix) == Config.DEBUG_SAMPLE_SIZE

            # Check dimensions based on model type
            if "anchor" in key:
                # MiniLM-L6-v2 -> 384 dims
                assert (
                    emb_matrix.shape[1] == 384
                ), f"Unexpected dim for {key}: {emb_matrix.shape[1]}"
            else:
                # MPNet-base-v2 -> 768 dims
                assert (
                    emb_matrix.shape[1] == 768
                ), f"Unexpected dim for {key}: {emb_matrix.shape[1]}"

    print("    Embeddings generated and verified.")

    # -------------------------------------------------------------------------
    # 4. Feature Pipeline (Transformation)
    # -------------------------------------------------------------------------
    print("\n[4] Running Feature Pipeline...")

    # Initialize Pipeline
    pipeline = FoldPipeline()

    # Prepare input dictionary for Training
    train_feats = {
        "anchor_title": embeddings["train"]["anchor_title"],
        "anchor_body": embeddings["train"]["anchor_body"],
        "aux_title": embeddings["train"]["aux_title"],
        "aux_body": embeddings["train"]["aux_body"],
        "meta": df_train,
    }

    # Fit Pipeline
    pipeline.fit(train_feats)
    assert pipeline.is_fitted, "Pipeline should be fitted after fit()"

    # Transform Training Data
    X_train = pipeline.transform(train_feats)

    # Calculate expected dimension:
    # 384 (Anchor Title) + 384 (Anchor Body) + 20 (Aux Title PCA) + 30 (Aux Body PCA) + 10 (Meta)
    # Note: Meta dimension depends on len(Config.NUMERICAL_FEATURES) which is 10 in config.py
    expected_dim = (
        384
        + 384
        + Config.AUX_TITLE_PCA_COMPONENTS
        + Config.AUX_BODY_PCA_COMPONENTS
        + len(Config.NUMERICAL_FEATURES)
    )

    assert X_train.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        expected_dim,
    ), f"X_train shape mismatch. Got {X_train.shape}, expected ({Config.DEBUG_SAMPLE_SIZE}, {expected_dim})"

    # Transform Validation Data
    val_feats = {k: embeddings["val"][k] for k in embeddings["val"]}
    val_feats["meta"] = df_val
    X_val = pipeline.transform(val_feats)
    assert X_val.shape == (Config.DEBUG_SAMPLE_SIZE, expected_dim)

    # Transform Test Data
    test_feats = {k: embeddings["test"][k] for k in embeddings["test"]}
    test_feats["meta"] = df_test
    X_test = pipeline.transform(test_feats)
    assert X_test.shape == (Config.DEBUG_SAMPLE_SIZE, expected_dim)

    print(f"    Feature transformation successful. Feature Vector Size: {expected_dim}")

    # Verify Pipeline Persistence
    pipe_path = os.path.join(Config.WORKING_DIR, "demo_pipeline.joblib")
    pipeline.save(pipe_path)
    loaded_pipe = FoldPipeline.load(pipe_path)
    assert loaded_pipe.is_fitted
    print("    Pipeline persistence verified.")

    # -------------------------------------------------------------------------
    # 5. Model Training
    # -------------------------------------------------------------------------
    print("\n[5] Training Ensemble Model...")

    y_train = df_train["requester_received_pizza"].values.astype(int)

    # Initialize Model Engine
    model_engine = EnsembleModel()

    # Train (includes Grid Search, though we restricted the grid)
    model_engine.optimize_and_train(X_train, y_train)

    assert model_engine.model is not None, "Model should be initialized after training"
    assert model_engine.best_params is not None, "Best params should be recorded"

    print(f"    Training complete. Best Params: {model_engine.best_params}")

    # Verify Model Persistence
    model_path = os.path.join(Config.WORKING_DIR, "demo_model.joblib")
    model_engine.save(model_path)

    loaded_engine = EnsembleModel()
    loaded_engine.load(model_path)
    assert loaded_engine.model is not None
    print("    Model persistence verified.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Predictions and Submission...")

    # Predict on Test Set
    probs = loaded_engine.predict_proba(X_test)

    # Verify Predictions
    assert len(probs) == Config.DEBUG_SAMPLE_SIZE
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be between 0 and 1"

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission/submission.csv")
    save_submission(df_test["request_id"].values, probs, submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect submission columns"
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), "Incorrect number of rows in submission"

    print(f"    Submission saved to {submission_path}")
    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
