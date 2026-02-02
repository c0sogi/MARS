import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import PawpularityDataset, get_transforms
from library.feature_extractor import MobileNetExtractor
from library.regressor import RidgeHead


def main():
    # 1. Initialization and Reproducibility
    print(">>> Step 1: Initialization")
    seed_everything(Config.SEED)

    # Define paths for demo outputs (using working directory)
    # We use specific filenames for this demo to avoid overwriting full-run caches if they exist
    demo_train_feats_path = os.path.join(Config.IDEA_DIR, "demo_train_features.npy")
    demo_train_targets_path = os.path.join(Config.IDEA_DIR, "demo_train_targets.npy")
    demo_val_feats_path = os.path.join(Config.IDEA_DIR, "demo_val_features.npy")
    demo_val_targets_path = os.path.join(Config.IDEA_DIR, "demo_val_targets.npy")
    demo_test_feats_path = os.path.join(Config.IDEA_DIR, "demo_test_features.npy")
    demo_test_ids_path = os.path.join(Config.IDEA_DIR, "demo_test_ids.npy")
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # 2. Data Loading
    print("\n>>> Step 2: Data Loading")
    # Get standard transforms
    transforms = get_transforms(Config.IMG_SIZE)

    # Initialize Datasets using provided metadata files
    train_dataset = PawpularityDataset(Config.TRAIN_METADATA, transform=transforms)
    val_dataset = PawpularityDataset(Config.VAL_METADATA, transform=transforms)
    test_dataset = PawpularityDataset(
        Config.TEST_METADATA, transform=transforms, test_mode=True
    )

    # Initialize DataLoaders
    # We use num_workers=2 to speed up loading without overwhelming the system
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 3. Feature Extraction
    print("\n>>> Step 3: Feature Extraction (MobileNetV3)")
    extractor = MobileNetExtractor()

    # To optimize for speed, we limit the number of batches processed.
    # This simulates the process on a subset of data.
    MAX_BATCHES = 5
    print(f"Extracting features for max {MAX_BATCHES} batches per loader...")

    # Extract Train Features
    train_feats, train_meta, train_targets = extractor.extract_features(
        train_loader,
        cache_features_path=demo_train_feats_path,
        cache_aux_path=demo_train_targets_path,
        load_cached_data=False,  # Force computation for demonstration
        is_test=False,
        max_batches=MAX_BATCHES,
    )

    # Verify Train Data
    assert train_feats.ndim == 2, "Feature matrix should be 2D"
    assert train_feats.shape[1] == 576, "MobileNetV3-Small output dim should be 576"
    assert train_meta.shape[1] == 12, "Metadata should have 12 features"
    assert len(train_feats) == len(train_targets), "Feature and target count mismatch"
    print(f"Train features shape: {train_feats.shape}")

    # Extract Validation Features
    val_feats, val_meta, val_targets = extractor.extract_features(
        val_loader,
        cache_features_path=demo_val_feats_path,
        cache_aux_path=demo_val_targets_path,
        load_cached_data=False,
        is_test=False,
        max_batches=MAX_BATCHES,
    )
    print(f"Val features shape: {val_feats.shape}")

    # 4. Model Training (Ridge Regression)
    print("\n>>> Step 4: Model Training")
    head = RidgeHead(alpha=1.0)

    # Fit the model
    # The regressor handles scaling and concatenation of image features + metadata internally
    head.fit(train_feats, train_meta, train_targets)

    # 5. Evaluation
    print("\n>>> Step 5: Evaluation")
    val_rmse = head.evaluate(val_feats, val_meta, val_targets)

    # Logic verification
    assert isinstance(val_rmse, float), "RMSE must be a float"
    assert val_rmse > 0, "RMSE must be positive"
    assert val_rmse < 100, "RMSE should be within reasonable bounds (0-100)"

    # 6. Inference and Submission
    print("\n>>> Step 6: Inference on Test Set")

    # Extract Test Features
    test_feats, test_meta, test_ids = extractor.extract_features(
        test_loader,
        cache_features_path=demo_test_feats_path,
        cache_aux_path=demo_test_ids_path,
        load_cached_data=False,
        is_test=True,
        max_batches=MAX_BATCHES,
    )

    # Predict
    test_preds = head.predict(test_feats, test_meta)

    # Verify Predictions
    assert len(test_preds) == len(test_ids), "Prediction count must match ID count"
    assert not np.isnan(test_preds).any(), "Predictions should not contain NaNs"

    # Save Submission
    head.save_submission(test_ids, test_preds, output_path=demo_submission_path)

    # Verify Output File
    assert os.path.exists(demo_submission_path), "Submission file was not created"
    df_sub = pd.read_csv(demo_submission_path)
    assert list(df_sub.columns) == [
        "Id",
        "Pawpularity",
    ], "Submission columns are incorrect"
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"

    print(f"\nDemo completed successfully. Submission saved to {demo_submission_path}")


if __name__ == "__main__":
    main()
