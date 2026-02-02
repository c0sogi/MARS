import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library components
from library.config import Config
from library.utils import setup_logger, set_seed, save_model, load_model
from library.data_loader import load_and_process_data
from library.embedding_manager import compute_embeddings, get_feature_schema
from library.pipeline_factory import create_model_pipeline


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")

    # Set a specific seed for reproducibility
    set_seed(42)

    # Configure Logger
    logger = setup_logger("demo_script", level=logging.INFO)

    # Override Config paths to use a demo directory
    # This prevents interference with the main 'working' directory
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.CACHE_TRAIN_FEATURES = os.path.join(demo_dir, "train_processed.parquet")
    Config.CACHE_VAL_FEATURES = os.path.join(demo_dir, "val_processed.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(demo_dir, "test_processed.parquet")
    Config.CACHE_TRAIN_EMBEDDINGS = os.path.join(demo_dir, "train_emb.npy")
    Config.CACHE_VAL_EMBEDDINGS = os.path.join(demo_dir, "val_emb.npy")
    Config.CACHE_TEST_EMBEDDINGS = os.path.join(demo_dir, "test_emb.npy")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission/submission.csv")

    # Override Hyperparameters for speed
    Config.N_BAGGING_ESTIMATORS = 2  # Reduced from 20
    Config.PARAM_GRID = {
        "estimator__C": [1.0],  # Single value to skip grid search overhead
        "estimator__class_weight": ["balanced"],
    }

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Data Loading...")

    # Load data (this reads from ./input and ./metadata)
    # We force reload to ensure we are testing the processing logic
    df_train, df_val, df_test = load_and_process_data(load_cached_data=False)

    print(f"Original Train Shape: {df_train.shape}")
    print(f"Original Val Shape: {df_val.shape}")
    print(f"Original Test Shape: {df_test.shape}")

    # Validate columns
    required_cols = [
        "request_id",
        Config.TEXT_COL_TITLE,
        Config.TEXT_COL_BODY,
    ] + Config.NUMERICAL_FEATURES
    for col in required_cols:
        if col not in df_train.columns:
            raise AssertionError(f"Missing required column in Train: {col}")

    # Create a small subset for rapid demonstration
    subset_size = 50
    df_train_sub = df_train.head(subset_size).copy()
    df_val_sub = df_val.head(subset_size).copy()
    df_test_sub = df_test.head(subset_size).copy()

    y_train_sub = df_train_sub["requester_received_pizza"].values.astype(int)
    y_val_sub = df_val_sub["requester_received_pizza"].values.astype(int)

    print(f"Subset Train Shape: {df_train_sub.shape}")

    # -------------------------------------------------------------------------
    # 3. Embedding Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Feature Extraction (Embeddings)...")

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models manually to process the subset
    # In a real run, generate_embeddings() does this, but we want to run on our subset
    print(f"Loading Anchor Model: {Config.MODEL_ANCHOR}")
    model_anchor = SentenceTransformer(Config.MODEL_ANCHOR, device=device)

    print(f"Loading Aux Model: {Config.MODEL_AUX}")
    model_aux = SentenceTransformer(Config.MODEL_AUX, device=device)

    # Compute embeddings for the subset
    print("Computing embeddings for training subset...")
    X_train_emb = compute_embeddings(df_train_sub, model_anchor, model_aux, device)

    print("Computing embeddings for validation subset...")
    X_val_emb = compute_embeddings(df_val_sub, model_anchor, model_aux, device)

    # Verify Embedding Dimensions
    # Schema: Title(384) + Body(384) + Global(768) + Meta(10) = 1546
    schema = get_feature_schema()
    expected_dim = schema["total_dims"]

    print(f"Computed Embedding Shape: {X_train_emb.shape}")
    print(f"Expected Dimension: {expected_dim}")

    if X_train_emb.shape[1] != expected_dim:
        raise AssertionError(
            f"Embedding dimension mismatch. Got {X_train_emb.shape[1]}, expected {expected_dim}"
        )

    # Clean up models to free memory
    del model_anchor
    del model_aux
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Pipeline Construction & Training Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Pipeline Construction and Training...")

    # Create the pipeline
    # We use a small PCA component count because our subset is small (50 samples)
    # PCA components must be <= min(n_samples, n_features)
    pipeline = create_model_pipeline(
        schema,
        pca_components=min(10, Config.PCA_COMPONENTS_AUX),
        n_estimators=Config.N_BAGGING_ESTIMATORS,
        base_params={"C": 0.1, "class_weight": "balanced"},
    )

    # Fit the pipeline
    print("Fitting pipeline on training subset...")
    pipeline.fit(X_train_emb, y_train_sub)

    # Predict on validation subset
    print("Predicting on validation subset...")
    val_probs = pipeline.predict_proba(X_val_emb)[:, 1]

    # Verify predictions
    if val_probs.shape[0] != subset_size:
        raise AssertionError(
            f"Prediction shape mismatch. Got {val_probs.shape[0]}, expected {subset_size}"
        )

    print(f"Validation Probabilities (First 5): {val_probs[:5]}")

    # Calculate Metric
    try:
        score = roc_auc_score(y_val_sub, val_probs)
        print(f"Subset AUC Score: {score:.4f}")
    except ValueError:
        # This might happen if the subset contains only one class
        print("Skipping AUC calculation (subset might contain only one class).")

    # -------------------------------------------------------------------------
    # 5. Model Persistence Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Model Persistence...")

    model_path = os.path.join(Config.WORKING_DIR, "demo_model.joblib")
    save_model(pipeline, model_path)
    print(f"Model saved to {model_path}")

    if not os.path.exists(model_path):
        raise AssertionError("Model file was not created.")

    # Reload model
    loaded_pipeline = load_model(model_path)
    print("Model reloaded successfully.")

    # Verify reloaded model predictions match
    reloaded_probs = loaded_pipeline.predict_proba(X_val_emb)[:, 1]
    if not np.allclose(val_probs, reloaded_probs):
        raise AssertionError("Reloaded model predictions do not match original model.")
    print("Reloaded model consistency check passed.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Submission Generation...")

    # We simulate the inference step using the test subset
    # Note: We need embeddings for the test subset first
    # Re-loading models just for this small test part (using CPU to save overhead if GPU busy, but here we use device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_anchor = SentenceTransformer(Config.MODEL_ANCHOR, device=device)
    model_aux = SentenceTransformer(Config.MODEL_AUX, device=device)

    X_test_emb = compute_embeddings(df_test_sub, model_anchor, model_aux, device)

    test_probs = loaded_pipeline.predict_proba(X_test_emb)[:, 1]

    submission_df = pd.DataFrame(
        {
            "request_id": df_test_sub["request_id"],
            "requester_received_pizza": test_probs,
        }
    )

    print(f"Submission DataFrame Head:\n{submission_df.head()}")

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    main()
