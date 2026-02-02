import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.dataset import get_dataset, PetDataset
from library.extractors import FeatureExtractor
from library.processors import DataProcessor
from library.models import ModelFactory
from library.engine import StackingTrainer


def main():
    print("=== Starting Library Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Patch Config for speed and low resource usage
    # Use a tiny ConvNeXt model instead of the large ones to save download/inference time
    Config.BACKBONES = ["facebook/convnext-tiny-224"]

    # Reduce CV folds
    Config.N_FOLDS = 2

    # Reduce hyperparameter search spaces and model complexity
    Config.RIDGE_ALPHAS = [0.1, 1.0, 10.0]
    Config.SVR_C = [1.0]  # Disable grid search for SVR
    Config.ET_N_ESTIMATORS = 10  # Fewer trees
    Config.PCA_COMPONENTS = (
        8  # Must be < number of samples in debug mode (approx 50 per fold)
    )
    Config.META_RIDGE_ALPHAS = [1.0]

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration patched successfully.")

    # -------------------------------------------------------------------------
    # 2. Utils Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utils...")
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 20.0, 28.0])
    rmse = compute_rmse(y_true, y_pred)
    # Expected: sqrt((4 + 0 + 4)/3) = sqrt(8/3) ~= 1.633
    assert np.isclose(rmse, np.sqrt(8 / 3)), f"RMSE calculation incorrect: {rmse}"
    print(f"Utils verified. RMSE: {rmse:.4f}")

    # -------------------------------------------------------------------------
    # 3. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset (Debug Mode)...")
    # Use debug=True to load only 100 samples
    train_ds = get_dataset("train", debug=True)

    assert (
        len(train_ds) == 100
    ), f"Expected 100 samples in debug mode, got {len(train_ds)}"

    # Fetch one sample
    img, meta, target, sample_id = train_ds[0]

    # Verify shapes and types
    # Image should be (C, H, W) -> (3, 224, 224) if transformed, but here we check raw loading logic mostly
    # The dataset class converts to PIL, but doesn't apply transforms unless passed.
    # However, the __getitem__ converts to PIL. If no transform, it returns PIL Image.
    # Wait, the provided Dataset code: "if self.transform: image = self.transform(image)".
    # If no transform is provided, it returns a PIL image.
    from PIL import Image

    assert isinstance(img, Image.Image), "Expected PIL Image when no transform provided"

    # Meta should be tensor of shape (12,)
    assert isinstance(meta, torch.Tensor)
    assert meta.shape == (12,), f"Expected metadata shape (12,), got {meta.shape}"

    # Target should be scalar tensor
    assert isinstance(target, torch.Tensor)
    assert target.ndim == 0, "Expected scalar target"

    print("Dataset verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Feature Extractor Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Feature Extractor...")
    # We use the patched backbone
    backbone = Config.BACKBONES[0]
    extractor = FeatureExtractor(backbone)

    # Run extraction on debug set (force re-computation by ignoring cache for this test)
    # We'll use a temporary cache dir or just let it overwrite since we are in working dir
    # Note: extract() saves to disk.
    features, ids, meta, targets = extractor.extract(
        "train", load_cached_data=False, debug=True
    )

    assert len(features) == 100, "Feature extraction count mismatch"
    assert len(ids) == 100
    assert len(meta) == 100
    assert len(targets) == 100

    # ConvNeXt Tiny has 768 dim output
    assert (
        features.shape[1] == 768
    ), f"Expected 768 feature dim, got {features.shape[1]}"

    print(f"Extraction verified. Feature shape: {features.shape}")

    # -------------------------------------------------------------------------
    # 5. Data Processor Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Data Processor...")
    processor = DataProcessor()

    # Test Linear Preparation (Scaling)
    X_lin, scaler = processor.prepare_linear_features(features, meta)
    # Shape should be (100, 768 + 12) = (100, 780)
    assert X_lin.shape == (
        100,
        768 + 12,
    ), f"Linear feature shape mismatch: {X_lin.shape}"
    assert np.abs(X_lin.mean()) < 1e-6, "StandardScaler failed to center data"

    # Test Tree Preparation (PCA)
    # We set PCA_COMPONENTS = 8 in config patch
    X_tree, pca = processor.prepare_tree_features(
        features, meta, n_components=Config.PCA_COMPONENTS
    )
    # Shape should be (100, 8 + 12) = (100, 20)
    assert X_tree.shape == (
        100,
        Config.PCA_COMPONENTS + 12,
    ), f"Tree feature shape mismatch: {X_tree.shape}"

    print("Data Processor verified.")

    # -------------------------------------------------------------------------
    # 6. Model Factory Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Model Experts...")
    factory = ModelFactory()

    # Test Linear Expert
    lin_model = factory.get_linear_expert()
    lin_model.fit(X_lin, targets)
    lin_preds = lin_model.predict(X_lin)
    assert lin_preds.shape == targets.shape

    # Test Partitioning Expert
    tree_model = factory.get_partitioning_expert()
    tree_model.fit(X_tree, targets)
    tree_preds = tree_model.predict(X_tree)
    assert tree_preds.shape == targets.shape

    print("Model experts verified (fit/predict successful).")

    # -------------------------------------------------------------------------
    # 7. End-to-End Engine Verification
    # -------------------------------------------------------------------------
    print("\n[7] Running End-to-End Stacking Trainer...")

    # Clean up any previous run artifacts in working directory to ensure fresh run
    # (Optional but good for verification)

    trainer = StackingTrainer()

    # Run full pipeline in debug mode
    # This will:
    # 1. Extract features for Train/Val/Test (using patched backbone)
    # 2. Run 2-Fold CV
    # 3. Train Meta-Learner
    # 4. Generate Submission
    trainer.run(load_cached_data=True, debug=True)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # In debug mode, test set is also limited to 100 rows (via dataset limit logic)
    # However, engine.py reads test.csv directly for IDs.
    # The engine.py run() method has a debug block: "if debug: test_df = test_df.iloc[:100]"
    # So we expect 100 rows.
    assert (
        len(sub_df) == 100
    ), f"Expected 100 rows in debug submission, got {len(sub_df)}"
    assert "Id" in sub_df.columns and "Pawpularity" in sub_df.columns

    print("\n=== Demonstration Complete: All Systems Operational ===")


if __name__ == "__main__":
    main()
