import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.feature_extraction import FeatureExtractor
from library.dimensionality_reduction import IndependentPCA
from library.models import get_base_models, get_meta_learner
from library.train_eval import merge_data_dicts, CrossValidator, FinalTrainer


def run_demo():
    print("Starting Pipeline Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Use a single lightweight backbone for speed
    Config.BACKBONES = ["tf_efficientnetv2_s"]

    # Reduce computational load
    Config.N_FOLDS = 2
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["verbose"] = -1
    Config.PCA_VARIANCE = 0.95  # Keep high to ensure we get some components

    # Re-run setup to create the new working directories
    Config.setup()
    seed_everything(Config.SEED)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Backbones: {Config.BACKBONES}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying Data Loading...")

    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Verify dataset sizes based on DEBUG truncation in data_loader.py
    # Train is truncated to 100, Val/Test to 50
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    assert len(train_loader.dataset) == 100, "Train dataset should be 100 in debug mode"
    assert len(val_loader.dataset) == 50, "Val dataset should be 50 in debug mode"

    # Verify batch structure
    batch = next(iter(train_loader))
    assert "image" in batch
    assert "metadata" in batch
    assert "target" in batch
    assert batch["image"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert batch["metadata"].shape == (Config.BATCH_SIZE, 12)  # 12 metadata features

    print("Data Loading verified successfully.")

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print("\n[3] Running Feature Extraction...")

    extractor = FeatureExtractor()

    # Force extraction (load_cached_data=False) to demonstrate the logic
    train_raw, val_raw, test_raw = extractor.extract_and_cache_features(
        load_cached_data=False
    )

    # Verify dictionary structure
    backbone_key = f"features_{Config.BACKBONES[0]}"

    assert backbone_key in train_raw
    assert "metadata" in train_raw
    assert "targets" in train_raw

    # Verify shapes
    n_train = len(train_loader.dataset)
    feat_dim = train_raw[backbone_key].shape[1]

    assert train_raw[backbone_key].shape == (n_train, feat_dim)
    assert train_raw["metadata"].shape == (n_train, 12)
    assert train_raw["targets"].shape == (n_train,)

    print(f"Extracted feature shape: {train_raw[backbone_key].shape}")
    print("Feature Extraction verified successfully.")

    # ==========================================
    # 4. Dimensionality Reduction (PCA)
    # ==========================================
    print("\n[4] Demonstrating Independent PCA...")

    pca_processor = IndependentPCA(
        variance_threshold=Config.PCA_VARIANCE, seed=Config.SEED
    )

    # Fit on training data
    pca_processor.fit(train_raw)

    # Transform validation data
    X_val_pca = pca_processor.transform(val_raw)

    # Check dimensions
    # Output dim = PCA components + 12 metadata features
    pca_components = pca_processor.pcas[Config.BACKBONES[0]].n_components_
    expected_dim = pca_components + 12

    print(
        f"Original Dim: {feat_dim} -> PCA Components: {pca_components} -> Total (+Meta): {expected_dim}"
    )
    assert X_val_pca.shape == (len(val_loader.dataset), expected_dim)

    print("PCA Transformation verified successfully.")

    # ==========================================
    # 5. Cross-Validation (Stacking Level 1)
    # ==========================================
    print("\n[5] Running Cross-Validation...")

    # Merge Train and Val for CV
    full_train_raw = merge_data_dicts(train_raw, val_raw)
    total_samples = len(full_train_raw["ids"])
    print(f"Merged Train+Val samples: {total_samples}")

    cv = CrossValidator(n_folds=Config.N_FOLDS, seed=Config.SEED)

    # Run CV (force computation, no cache loading)
    oof_preds_df, targets = cv.run_cv(full_train_raw, load_cached_data=False)

    # Verify OOF output
    base_models = get_base_models()
    assert oof_preds_df.shape == (total_samples, len(base_models))
    assert list(oof_preds_df.columns) == list(base_models.keys())
    assert len(targets) == total_samples

    # Check for NaNs
    assert not oof_preds_df.isnull().values.any(), "OOF predictions contain NaNs"

    print("Cross-Validation verified successfully.")

    # ==========================================
    # 6. Final Training & Submission
    # ==========================================
    print("\n[6] Running Final Training & Inference...")

    trainer = FinalTrainer(seed=Config.SEED)

    submission = trainer.train_and_predict(
        full_train_raw, test_raw, oof_preds_df, targets
    )

    # Verify Submission
    assert isinstance(submission, pd.DataFrame)
    assert submission.shape == (len(test_loader.dataset), 2)
    assert list(submission.columns) == ["Id", "Pawpularity"]

    # Verify value range (1-100)
    preds = submission["Pawpularity"].values
    assert (preds >= 1.0).all() and (
        preds <= 100.0
    ).all(), "Predictions out of range [1, 100]"

    # Verify file existence
    assert os.path.exists(Config.SUBMISSION_PATH)

    print("\n=== Demo Completed Successfully ===")
    print(f"Sample Submission:\n{submission.head()}")


if __name__ == "__main__":
    run_demo()
