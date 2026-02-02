import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, rmse_score
from library.data import load_metadata, PetDataset, get_processor
from library.processors import LogitTargetTransformer, FeaturePreprocessor
from library.extractors import process_and_cache_features
from library.ensemble import Level0Trainer, Level1Trainer, run_ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 0. Configuration & Setup
    # -------------------------------------------------------------------------
    # Modify Config at runtime to ensure speed and compatibility with small debug data
    print("\n[Step 0] Configuring environment...")
    Config.N_FOLDS = 2  # Reduce folds for speed
    Config.DEBUG_SAMPLE_SIZE = 30  # Small sample size for demonstration
    Config.IDEA_DIR = "./working/demo_execution"  # Separate working dir for demo

    # Ensure clean state
    if os.path.exists(Config.IDEA_DIR):
        shutil.rmtree(Config.IDEA_DIR)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration patched: N_FOLDS=2, DEBUG_SAMPLE_SIZE=30")

    # -------------------------------------------------------------------------
    # 1. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying Data Loading...")
    # Load debug subset of metadata
    train_df, val_df, test_df = load_metadata(
        merge_train_val=False, debug=True, sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify shapes
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} train samples"
    assert (
        len(test_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} test samples"
    assert "file_path" in train_df.columns
    assert "Pawpularity" in train_df.columns
    print("Data loading verified successfully.")

    # -------------------------------------------------------------------------
    # 2. Dataset & Image Processor
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Dataset and Processor...")
    # Use ConvNeXt for demonstration as it is one of the backbones
    model_name = Config.MODEL_CONVNEXTV2
    processor = get_processor(model_name)

    # Instantiate Dataset with return_flipped=True to test augmentation logic
    ds = PetDataset(train_df, processor, return_flipped=True, include_target=True)

    # Verify item structure
    sample = ds[0]
    assert "pixel_values" in sample
    assert "metadata" in sample
    assert "target" in sample
    assert "id" in sample

    # Check shapes: (2, C, H, W) because return_flipped=True
    # ConvNeXt V2 typically uses 3 channels, 224x224
    pixel_shape = sample["pixel_values"].shape
    assert pixel_shape[0] == 2, f"Expected 2 views (flipped), got {pixel_shape[0]}"
    assert pixel_shape[1] == 3, "Expected 3 channels"
    print(f"Dataset verified. Image shape (with flip): {pixel_shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Feature Extraction...")
    # Define cache paths for this demo step
    cache_paths = {
        "ids": os.path.join(Config.IDEA_DIR, "demo_ids.npy"),
        "embeddings": os.path.join(Config.IDEA_DIR, "demo_emb.npy"),
        "metadata": os.path.join(Config.IDEA_DIR, "demo_meta.npy"),
        "targets": os.path.join(Config.IDEA_DIR, "demo_tgt.npy"),
    }

    # Run extraction
    # We use a small batch size for the demo
    data_dict = process_and_cache_features(
        ds, model_name, batch_size=8, cache_paths=cache_paths, load_cached_data=False
    )

    # Verify outputs
    embeddings = data_dict["embeddings"]
    targets = data_dict["targets"]
    metadata = data_dict["metadata"]
    ids = data_dict["ids"]

    assert len(embeddings) == Config.DEBUG_SAMPLE_SIZE
    assert len(targets) == Config.DEBUG_SAMPLE_SIZE
    # Embeddings should be 2D: (N, D)
    assert embeddings.ndim == 2
    print(f"Feature extraction verified. Embedding shape: {embeddings.shape}")

    # -------------------------------------------------------------------------
    # 4. Preprocessing Logic
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Preprocessors...")

    # A. Target Transformation
    tgt_transformer = LogitTargetTransformer()
    # Test values
    y_raw = np.array([1.0, 50.0, 99.0])
    y_trans = tgt_transformer.transform(y_raw)
    y_recon = tgt_transformer.inverse_transform(y_trans)

    assert np.all(y_trans > -np.inf) and np.all(
        y_trans < np.inf
    ), "Transformed targets contain infinity"
    assert np.allclose(y_raw, y_recon, atol=1e-3), "Target reconstruction failed"
    print("Target transformation verified.")

    # B. Feature Preprocessing
    # Test 'tree' strategy (PCA on embeddings + raw metadata)
    # Reduce PCA components to be valid for small sample size
    preprocessor = FeaturePreprocessor(
        pca_components=min(10, Config.DEBUG_SAMPLE_SIZE - 1)
    )
    X_tree = preprocessor.fit_transform(embeddings, metadata, strategy="tree")

    expected_dim = preprocessor.pca_components + 12  # 12 metadata features
    assert X_tree.shape == (Config.DEBUG_SAMPLE_SIZE, expected_dim)
    print("Feature preprocessing (Tree strategy) verified.")

    # -------------------------------------------------------------------------
    # 5. Level 0 Training (Base Models)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Level 0 Training...")

    # Construct input dictionaries as expected by Level0Trainer
    # We'll use the same extracted features for 'train' and 'test' just to verify flow
    feature_data_mock = {"ConvNeXt": {"embeddings": embeddings, "metadata": metadata}}
    test_data_mock = {"ConvNeXt": {"embeddings": embeddings, "metadata": metadata}}

    l0_trainer = Level0Trainer(n_folds=Config.N_FOLDS)

    # Run CV
    oof_preds, l0_test_preds = l0_trainer.run_cv(
        feature_data_mock, targets, test_data_mock, load_cached=False
    )

    # Expected shape: (N_samples, N_backbones * N_models)
    # Backbones=1, Models=4 (Ridge, SVR, ET, LGBM) -> 4 columns
    assert oof_preds.shape == (Config.DEBUG_SAMPLE_SIZE, 4)
    assert l0_test_preds.shape == (Config.DEBUG_SAMPLE_SIZE, 4)

    # Check RMSE of OOF to ensure models learned something (or at least didn't explode)
    # With 30 samples, performance will be poor, but should be valid numbers
    score = rmse_score(targets, oof_preds[:, 0])  # Check first model
    assert not np.isnan(score)
    print(
        f"Level 0 Training verified. OOF Shape: {oof_preds.shape}, RMSE (Model 1): {score:.4f}"
    )

    # -------------------------------------------------------------------------
    # 6. Level 1 Training (Meta Learner)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Level 1 Training...")

    l1_trainer = Level1Trainer()
    final_preds = l1_trainer.train_and_predict(oof_preds, targets, l0_test_preds)

    assert final_preds.shape == (Config.DEBUG_SAMPLE_SIZE,)
    assert not np.isnan(final_preds).any()
    print("Level 1 Training verified.")

    # -------------------------------------------------------------------------
    # 7. Full Integration Test
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Full Ensemble Pipeline (Integration Test)...")

    # We override the cache paths in Config to point to our temp dir
    # This ensures run_ensemble writes to our disposable directory
    Config.IDEA_DIR = os.path.join(Config.WORKING_DIR, "demo_integration")
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Update cache paths in Config to use this new dir
    Config.CACHE_FEATURES_SIGLIP = os.path.join(Config.IDEA_DIR, "features_siglip.npy")
    Config.CACHE_FEATURES_DINOV2 = os.path.join(Config.IDEA_DIR, "features_dinov2.npy")
    Config.CACHE_FEATURES_CONVNEXT = os.path.join(
        Config.IDEA_DIR, "features_convnext.npy"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "submission.csv")

    # Run the full pipeline with debug=True
    # This will:
    # 1. Load data (subset)
    # 2. Extract features for ALL 3 backbones (SigLIP, DINOv2, ConvNeXt)
    # 3. Train Level 0 (4 models * 3 backbones = 12 models)
    # 4. Train Level 1
    # 5. Generate submission
    run_ensemble(debug=True, load_cached_level0=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_df) == Config.DEBUG_SAMPLE_SIZE
    assert "Id" in submission_df.columns
    assert "Pawpularity" in submission_df.columns

    print("\n=== Demonstration Complete ===")
    print(f"Submission generated at: {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
