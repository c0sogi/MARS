import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.data_factory import get_loader
from library.backbone_extractor import BackboneExtractor
from library.feature_processor import FeatureProcessor
from library.stacking_engine import Level0Trainer, Level1MetaLearner

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    print("=== Starting Demonstration of Pet Pawpularity Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Runtime Configuration Overrides (Optimize for Speed)
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config to run fast on a small subset
    Config.DEBUG = True
    Config.N_FOLDS = 2  # Minimal folds for CV
    Config.NUM_WORKERS = 2

    # Reduce hyperparameter grids to single values to skip tuning time
    Config.RIDGE_ALPHAS = [1.0]
    Config.SVR_GRID = {"kernel": ["rbf"], "C": [1.0], "epsilon": [0.1]}
    Config.ET_PARAMS = {
        "n_estimators": 10,
        "n_jobs": -1,
        "random_state": Config.SEED,
        "verbose": 0,
    }
    Config.ET_GRID = {"max_depth": [5], "min_samples_leaf": [1]}
    Config.META_MODEL_PARAMS = {"max_iter": 10, "verbose": False, "compute_score": True}

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Clean up working directory for this demo to ensure fresh run
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Validation
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading (data_factory.py)...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Subset for speed (20 train, 10 test)
    train_subset = train_df.head(20).reset_index(drop=True)
    test_subset = test_df.head(10).reset_index(drop=True)

    print(f"Training subset size: {len(train_subset)}")
    print(f"Test subset size: {len(test_subset)}")

    # Test DataLoader with 'convnext' backbone
    backbone_name = "convnext"
    loader = get_loader(
        df=train_subset,
        backbone_name=backbone_name,
        batch_size=4,
        shuffle=False,
        return_flip=True,  # Test augmentation flag
    )

    # Fetch one batch to verify structure
    batch = next(iter(loader))

    # Assertions
    assert "pixel_values" in batch, "Batch missing pixel_values"
    assert (
        "pixel_values_flip" in batch
    ), "Batch missing pixel_values_flip (return_flip=True)"
    assert "meta_features" in batch, "Batch missing meta_features"
    assert "target" in batch, "Batch missing target"
    assert batch["pixel_values"].shape[0] == 4, "Batch size mismatch"
    # ConvNeXt resolution is 224
    assert batch["pixel_values"].shape[2:] == (
        224,
        224,
    ), "Image resolution mismatch for ConvNeXt"

    print("DataLoader verification successful.")

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print(
        f"\n[3] Demonstrating Feature Extraction for '{backbone_name}' (backbone_extractor.py)..."
    )

    extractor = BackboneExtractor()

    # Extract Train Features (force recompute by setting load_cached_data=False initially or relying on cleared dir)
    # We pass the subset dataframe, so the extractor processes only these 20 images.
    train_data = extractor.extract(
        df=train_subset,
        backbone_name=backbone_name,
        subset_name="train_demo",
        load_cached_data=False,
    )

    # Extract Test Features
    test_data = extractor.extract(
        df=test_subset,
        backbone_name=backbone_name,
        subset_name="test_demo",
        load_cached_data=False,
    )

    # Assertions
    expected_dim = Config.BACKBONES[backbone_name]["output_dim"]
    assert train_data["features"].shape == (
        20,
        expected_dim,
    ), f"Train features shape mismatch. Expected (20, {expected_dim})"
    assert train_data["targets"].shape == (20,), "Train targets shape mismatch"
    assert test_data["features"].shape == (
        10,
        expected_dim,
    ), f"Test features shape mismatch. Expected (10, {expected_dim})"

    print("Feature extraction successful.")

    # -------------------------------------------------------------------------
    # 4. Feature Processing
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Feature Processing (feature_processor.py)...")

    processor = FeatureProcessor()

    # Test Linear Processing (StandardScaler)
    # Fit on train, transform train
    X_lin_train = processor.prepare_data_for_linear(
        train_data["features"], train_data["meta"], fit=True
    )
    # Transform test
    X_lin_test = processor.prepare_data_for_linear(
        test_data["features"], test_data["meta"], fit=False
    )

    # Assertions for Linear
    # Output dim = Embedding Dim + Metadata Dim (12)
    expected_lin_dim = expected_dim + 12
    assert X_lin_train.shape == (20, expected_lin_dim), "Linear feature shape mismatch"
    assert processor.linear_fitted is True, "Scaler should be fitted"

    # Test Tree Processing (PCA)
    # Fit on train, transform train
    X_tree_train = processor.prepare_data_for_tree(
        train_data["features"], train_data["meta"], fit=True
    )

    # Assertions for Tree
    # Output dim = PCA Components (clamped) + Metadata Dim (12)
    expected_tree_dim = processor.pca.n_components + 12
    assert X_tree_train.shape == (20, expected_tree_dim), "Tree feature shape mismatch"
    assert processor.tree_fitted is True, "PCA should be fitted"

    print("Feature processing verification successful.")

    # -------------------------------------------------------------------------
    # 5. Level-0 Stacking (Expert Training)
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Level-0 Expert Training (stacking_engine.py)...")

    l0_trainer = Level0Trainer()

    # Dictionary to store OOF and Test predictions for Level 1
    oof_preds_dict = {}
    test_preds_dict = {}

    # Run Ridge Expert
    print("Running Ridge Expert...")
    oof_ridge, test_ridge = l0_trainer.run_expert(
        backbone_name=backbone_name,
        model_type="ridge",
        train_data=train_data,
        test_data=test_data,
        load_cached_data=False,
    )
    oof_preds_dict[f"{backbone_name}_ridge"] = oof_ridge
    test_preds_dict[f"{backbone_name}_ridge"] = test_ridge

    # Run ExtraTrees Expert
    print("Running ExtraTrees Expert...")
    oof_et, test_et = l0_trainer.run_expert(
        backbone_name=backbone_name,
        model_type="et",
        train_data=train_data,
        test_data=test_data,
        load_cached_data=False,
    )
    oof_preds_dict[f"{backbone_name}_et"] = oof_et
    test_preds_dict[f"{backbone_name}_et"] = test_et

    # Assertions
    assert len(oof_ridge) == 20, "Ridge OOF length mismatch"
    assert len(test_ridge) == 10, "Ridge Test pred length mismatch"
    assert not np.isnan(oof_ridge).any(), "Ridge OOF contains NaNs"

    print("Level-0 training successful.")

    # -------------------------------------------------------------------------
    # 6. Level-1 Meta-Learner
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Level-1 Meta-Learner (stacking_engine.py)...")

    l1_learner = Level1MetaLearner()

    # Targets for meta-learner training
    y_true = train_data["targets"]
    test_ids = test_data["ids"]

    # Train and Predict
    final_preds = l1_learner.train_and_predict(
        oof_dict=oof_preds_dict,
        test_pred_dict=test_preds_dict,
        y_true=y_true,
        test_ids=test_ids,
    )

    # Assertions
    assert len(final_preds) == 10, "Final predictions length mismatch"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Validate submission file format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["Id", "Pawpularity"], "Submission columns mismatch"
    assert len(sub_df) == 10, "Submission row count mismatch"

    print(f"Submission generated at: {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
