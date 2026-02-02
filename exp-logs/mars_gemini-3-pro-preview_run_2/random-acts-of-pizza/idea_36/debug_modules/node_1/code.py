import os
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Import library components
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_dataset
from library.embeddings import EmbeddingGenerator
from library.feature_engineering import ContextAwareFusionTransformer, assemble_features
from library.model_trainer import train_model


def setup_demo_environment():
    """
    Overrides Config parameters for a fast, self-contained demonstration.
    """
    print(">>> Setting up demo configuration...")

    # Enable Debug mode to limit data size
    Config.DEBUG = True
    Config.MAX_SAMPLES = 50  # Process only 50 samples for speed

    # Reduce computational complexity for demo
    Config.N_FOLDS = 2
    Config.N_BAGGING_ESTIMATORS = 2

    # Simplify Grid Search to a single parameter to skip time-consuming tuning
    Config.PARAM_GRID = {
        "base_estimator__C": [1.0],
        "base_estimator__class_weight": ["balanced"],
    }

    # Redirect paths to a demo directory within ./working
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Update cache paths to point to the demo directory
    # This ensures we don't load any pre-existing full-scale data
    Config.TRAIN_EMB_ANCHOR_PATH = os.path.join(demo_dir, "demo_train_emb_anchor.npy")
    Config.TRAIN_EMB_AUX_PATH = os.path.join(demo_dir, "demo_train_emb_aux.npy")
    Config.VAL_EMB_ANCHOR_PATH = os.path.join(demo_dir, "demo_val_emb_anchor.npy")
    Config.VAL_EMB_AUX_PATH = os.path.join(demo_dir, "demo_val_emb_aux.npy")
    Config.TEST_EMB_ANCHOR_PATH = os.path.join(demo_dir, "demo_test_emb_anchor.npy")
    Config.TEST_EMB_AUX_PATH = os.path.join(demo_dir, "demo_test_emb_aux.npy")

    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "demo_train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(demo_dir, "demo_val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "demo_test_features.parquet")

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Max Samples: {Config.MAX_SAMPLES}")
    print(f"Working Directory: {Config.WORKING_DIR}")


def demonstrate_data_loading():
    """
    Demonstrates loading the dataset and verifies the subsetting logic.
    """
    print("\n>>> Demonstrating Data Loading...")

    # Load training data
    df_train = load_dataset("train", load_cached_data=False)

    # Verify dimensions
    print(f"Loaded Train DataFrame Shape: {df_train.shape}")

    if Config.DEBUG:
        assert (
            len(df_train) == Config.MAX_SAMPLES
        ), f"Expected {Config.MAX_SAMPLES} samples in DEBUG mode, got {len(df_train)}"

    # Verify essential columns
    expected_cols = [Config.ID_COL, "text_concat"] + Config.NUMERIC_COLS
    for col in expected_cols:
        assert col in df_train.columns, f"Missing expected column: {col}"

    print("Data loading verification passed.")


def demonstrate_embeddings():
    """
    Demonstrates embedding generation for Anchor and Aux views.
    """
    print("\n>>> Demonstrating Embedding Generation...")

    generator = EmbeddingGenerator()

    # Generate Anchor Embeddings (MiniLM-L6-v2 -> 384d)
    # Note: This will trigger model loading/downloading if not cached
    emb_anchor = generator.get_embeddings("train", "anchor", load_cached_data=False)
    print(f"Anchor Embeddings Shape: {emb_anchor.shape}")

    assert emb_anchor.shape == (
        Config.MAX_SAMPLES,
        384,
    ), f"Expected shape ({Config.MAX_SAMPLES}, 384), got {emb_anchor.shape}"

    # Generate Aux Embeddings (MPNet-Base-v2 -> 768d)
    emb_aux = generator.get_embeddings("train", "aux", load_cached_data=False)
    print(f"Aux Embeddings Shape: {emb_aux.shape}")

    assert emb_aux.shape == (
        Config.MAX_SAMPLES,
        768,
    ), f"Expected shape ({Config.MAX_SAMPLES}, 768), got {emb_aux.shape}"

    print("Embedding generation verification passed.")


def demonstrate_feature_assembly():
    """
    Demonstrates assembling all feature views into a single matrix.
    """
    print("\n>>> Demonstrating Feature Assembly...")

    # Assemble features for training split
    X, y = assemble_features("train", load_cached_data=True)

    print(f"Assembled Matrix X Shape: {X.shape}")
    print(f"Target Vector y Shape: {y.shape}")

    # Expected Width: Anchor(384) + Aux(768) + Meta(10) = 1162
    expected_width = 384 + 768 + len(Config.NUMERIC_COLS)
    assert (
        X.shape[1] == expected_width
    ), f"Expected feature width {expected_width}, got {X.shape[1]}"

    assert len(X) == len(y) == Config.MAX_SAMPLES, "Sample count mismatch."

    print("Feature assembly verification passed.")
    return X, y


def demonstrate_transformer(X_train):
    """
    Demonstrates the ContextAwareFusionTransformer logic.
    """
    print("\n>>> Demonstrating Context-Aware Fusion Transformer...")

    # Initialize transformer
    # We use reduced PCA components for the demo if necessary, but Config default is 50
    transformer = ContextAwareFusionTransformer(
        anchor_dim=384,
        aux_dim=768,
        pca_components=Config.AUX_PCA_COMPONENTS,
        interaction_top_k=Config.INTERACTION_TOP_K,
        random_state=Config.SEED,
    )

    # Fit and Transform
    print("Fitting transformer...")
    X_fused = transformer.fit_transform(X_train)

    print(f"Fused Feature Matrix Shape: {X_fused.shape}")

    # calculate expected output dimension
    # 1. Anchor Norm: 384
    # 2. Aux PCA Norm: 50 (Config.AUX_PCA_COMPONENTS)
    # 3. Meta Scaled: 10 (len(Config.NUMERIC_COLS))
    # 4. Interactions:
    #    Inputs = TopK(5) + Meta(10) = 15
    #    Poly(degree=2, interaction_only=True) includes linear terms (degree 1) and cross-products (degree 2)
    #    Linear terms = 15
    #    Cross-products = n(n-1)/2 = 15*14/2 = 105
    #    Total Interactions View = 15 + 105 = 120
    # Total = 384 + 50 + 10 + 120 = 564

    n_meta = len(Config.NUMERIC_COLS)
    n_inter_input = Config.INTERACTION_TOP_K + n_meta

    n_linear = n_inter_input
    n_cross_product = (n_inter_input * (n_inter_input - 1)) // 2
    n_interactions_total = n_linear + n_cross_product

    expected_dim = 384 + Config.AUX_PCA_COMPONENTS + n_meta + n_interactions_total

    assert (
        X_fused.shape[1] == expected_dim
    ), f"Expected fused dimension {expected_dim}, got {X_fused.shape[1]}"

    # Check for NaNs
    assert not np.isnan(X_fused).any(), "Fused features contain NaNs."

    print("Transformer verification passed.")


def demonstrate_training_pipeline():
    """
    Runs the full training pipeline (CV -> Inference -> Submission).
    """
    print("\n>>> Running Full Training Pipeline...")

    # This function handles data loading, CV, model fitting, and submission generation
    oof_auc = train_model(load_cached_data=True)

    print(f"Pipeline completed. OOF AUC: {oof_auc:.4f}")

    # Verify Submission File
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission File Shape: {df_sub.shape}")
    print(f"Submission Head:\n{df_sub.head()}")

    # Verify columns
    assert Config.ID_COL in df_sub.columns, "Missing ID column in submission"
    assert Config.TARGET_COL in df_sub.columns, "Missing Target column in submission"

    # Verify value range
    preds = df_sub[Config.TARGET_COL]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    # Verify row count matches test set (subset in debug mode)
    # Note: train_model loads test data with Config.DEBUG, so it should be MAX_SAMPLES
    assert (
        len(df_sub) == Config.MAX_SAMPLES
    ), f"Expected {Config.MAX_SAMPLES} predictions, got {len(df_sub)}"

    print("Training pipeline and submission verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Setup
    setup_demo_environment()

    # Step 1: Data Loading
    demonstrate_data_loading()

    # Step 2: Embeddings
    demonstrate_embeddings()

    # Step 3: Feature Assembly
    X_train, y_train = demonstrate_feature_assembly()

    # Step 4: Transformer Logic
    demonstrate_transformer(X_train)

    # Step 5: Full Pipeline
    demonstrate_training_pipeline()

    print("\n>>> All demonstrations completed successfully.")
