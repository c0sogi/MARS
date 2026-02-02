import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from functools import partialmethod
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_rmse, create_stratified_folds
from library.dataset import PawpularityDataset, get_dataloader
from library.feature_extraction import DeepFeatureExtractor
from library.feature_engineering import FeatureEngineer
from library.models import (
    get_ridge_expert,
    get_svr_expert,
    get_extratrees_expert,
    get_lgbm_expert,
    get_meta_learner,
)
from library.workflow import StratifiedStackingRunner
from transformers import AutoImageProcessor

# Suppress tqdm progress bars
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def main():
    print("Starting demonstration of Pawpularity solution components...")

    # =========================================================================
    # 0. Configuration Override for Speed
    # =========================================================================
    print("\n[0] Configuring environment for fast execution...")

    # Set global config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small subset
    Config.N_FOLDS = 2  # Minimal folds
    Config.STRATIFY_BINS = 5  # Fewer bins for small sample size

    # Use only one backbone to save download/inference time
    # ConvNeXt is generally efficient and effective
    Config.BACKBONES = {"convnext": Config.BACKBONES["convnext"]}

    # Simplify Model Hyperparameters
    Config.PCA_COMPONENTS = 10  # Reduced for small sample size
    Config.RIDGE_ALPHAS = [1.0]  # No search
    Config.SVR_PARAMS = {  # Minimal grid
        "kernel": "rbf",
        "C": [1.0],
        "epsilon": [0.1],
        "cache_size": 200,
    }
    Config.ET_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.META_MODEL_PARAMS["max_iter"] = 10

    # Ensure working directory is clean for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    seed_everything(Config.SEED)
    print("Configuration updated.")

    # =========================================================================
    # 1. Demonstrate Utils
    # =========================================================================
    print("\n[1] Demonstrating library.utils...")

    # Load a small sample of metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    # Test Stratified Folds
    folds = create_stratified_folds(
        df_train, n_folds=Config.N_FOLDS, n_bins=Config.STRATIFY_BINS
    )

    assert (
        len(folds) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} folds, got {len(folds)}"
    assert len(folds[0]) == 2, "Fold tuple should contain (train_idx, val_idx)"
    print(
        f"Stratified folds created successfully. Fold 0 Train size: {len(folds[0][0])}"
    )

    # =========================================================================
    # 2. Demonstrate Dataset
    # =========================================================================
    print("\n[2] Demonstrating library.dataset...")

    # Initialize processor for the chosen backbone
    backbone_cfg = Config.BACKBONES["convnext"]
    processor = AutoImageProcessor.from_pretrained(backbone_cfg["model_id"])

    # Instantiate Dataset
    dataset = PawpularityDataset(
        dataframe=df_train,
        processor=processor,
        root_dir=Config.INPUT_DIR,
        is_train=True,
        return_flipped=True,
    )

    # Test __getitem__
    sample = dataset[0]

    # Assertions
    assert "pixel_values" in sample
    assert "pixel_values_flipped" in sample
    assert "features" in sample
    assert "label" in sample
    assert "Id" in sample

    # Check shapes (ConvNeXt uses 224x224)
    expected_shape = (3, 224, 224)
    assert (
        sample["pixel_values"].shape == expected_shape
    ), f"Image shape mismatch: {sample['pixel_values'].shape}"
    assert isinstance(sample["label"], torch.Tensor)

    print("Dataset loaded and processed a sample successfully.")

    # =========================================================================
    # 3. Demonstrate Feature Extraction
    # =========================================================================
    print("\n[3] Demonstrating library.feature_extraction...")

    extractor = DeepFeatureExtractor()

    # Run extraction on the small dataframe
    # This will use the cache logic, but since we cleared working dir, it will compute.
    extracted_data = extractor.extract_features(
        dataframe=df_train,
        backbone_key="convnext",
        subset_name="demo_train",
        load_cached_data=False,
    )

    features = extracted_data["features"]
    meta = extracted_data["meta"]
    targets = extracted_data["targets"]

    # Assertions
    assert features.shape[0] == Config.DEBUG_SAMPLE_SIZE
    # ConvNeXt Large usually has 1536 dim features (pooler_output)
    # Check if features are 2D
    assert len(features.shape) == 2
    assert meta.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert meta.shape[1] == 12  # 12 binary features
    assert targets.shape[0] == Config.DEBUG_SAMPLE_SIZE

    print(f"Features extracted: {features.shape}, Meta: {meta.shape}")

    # =========================================================================
    # 4. Demonstrate Feature Engineering
    # =========================================================================
    print("\n[4] Demonstrating library.feature_engineering...")

    engineer = FeatureEngineer()

    # 4a. Linear Features (StandardScaler)
    # We simulate a 'train' split to fit the scaler
    X_linear = engineer.prepare_linear_features(
        features, meta, "convnext", "train", load_cached_data=False
    )

    # Shape should be embedding_dim + meta_dim
    expected_dim_linear = features.shape[1] + meta.shape[1]
    assert X_linear.shape == (Config.DEBUG_SAMPLE_SIZE, expected_dim_linear)

    # 4b. Tree Features (PCA)
    # We simulate a 'train' split to fit PCA
    X_tree = engineer.prepare_tree_features(
        features, meta, "convnext", "train", load_cached_data=False
    )

    # Shape should be pca_components + meta_dim
    expected_dim_tree = Config.PCA_COMPONENTS + meta.shape[1]
    assert X_tree.shape == (Config.DEBUG_SAMPLE_SIZE, expected_dim_tree)

    print("Feature engineering (Linear & Tree) completed successfully.")

    # =========================================================================
    # 5. Demonstrate Models
    # =========================================================================
    print("\n[5] Demonstrating library.models...")

    y = targets

    # 5a. Ridge Expert
    print("Training Ridge Expert...")
    ridge = get_ridge_expert()
    ridge.fit(X_linear, y)
    preds_ridge = ridge.predict(X_linear)
    rmse_ridge = calculate_rmse(y, preds_ridge)
    assert preds_ridge.shape[0] == Config.DEBUG_SAMPLE_SIZE
    print(f"Ridge RMSE: {rmse_ridge:.4f}")

    # 5b. LightGBM Expert
    print("Training LightGBM Expert...")
    lgbm = get_lgbm_expert({"verbosity": -1})
    lgbm.fit(X_tree, y)
    preds_lgbm = lgbm.predict(X_tree)
    rmse_lgbm = calculate_rmse(y, preds_lgbm)
    assert preds_lgbm.shape[0] == Config.DEBUG_SAMPLE_SIZE
    print(f"LightGBM RMSE: {rmse_lgbm:.4f}")

    # =========================================================================
    # 6. Demonstrate Full Workflow
    # =========================================================================
    print("\n[6] Demonstrating library.workflow (StratifiedStackingRunner)...")

    # Initialize runner in debug mode
    runner = StratifiedStackingRunner(debug=True)

    # Run the pipeline
    # This will:
    # 1. Load metadata
    # 2. Extract features (using cached convnext from step 3 if keys match, or re-compute)
    # 3. Run CV with 2 folds
    # 4. Train Meta Learner
    # 5. Predict on Test
    # 6. Save submission
    runner.run()

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print(sub_df.head())

        # Verify submission length matches debug sample size
        assert len(sub_df) == Config.DEBUG_SAMPLE_SIZE
        assert "Id" in sub_df.columns
        assert "Pawpularity" in sub_df.columns
    else:
        raise FileNotFoundError("Submission file was not created by the runner.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
