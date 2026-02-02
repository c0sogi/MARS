import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from library
from library.config import Config
from library.utils import seed_everything, rmse_score
from library.data_loader import get_dataloader, PetDataset
from library.feature_extractor import run_extraction, FeatureExtractor
from library.ensemble_components import get_linear_expert, get_partitioning_expert
from library.cross_validation import CrossValidator


def demo_data_loading():
    print("\n=== Demo 1: Data Loading ===")
    # Demonstrate Dataset and DataLoader instantiation with debug=True (small subset)
    print("Initializing DataLoader (Debug Mode)...")
    # We use a small batch size for demonstration
    loader = get_dataloader(
        csv_path=Config.TRAIN_METADATA_PATH,
        batch_size=8,
        view_mode="warped",
        backbone_type="dinov2",
        shuffle=True,
        debug=True,  # Loads only 64 samples
    )

    print(f"DataLoader created. Number of batches: {len(loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(loader))
    images = batch["image"]
    meta = batch["meta"]
    targets = batch["target"]
    ids = batch["id"]

    print(
        f"Batch shapes - Images: {images.shape}, Meta: {meta.shape}, Targets: {targets.shape}"
    )

    # Assertions to verify correctness
    # Image size should be (Batch, 3, 224, 224)
    assert images.shape == (
        8,
        3,
        224,
        224,
    ), f"Image tensor shape mismatch: {images.shape}"
    # Metadata should be (Batch, 12)
    assert meta.shape == (8, 12), f"Metadata tensor shape mismatch: {meta.shape}"
    # Targets should be (Batch,)
    assert targets.shape == (8,), f"Target tensor shape mismatch: {targets.shape}"
    # IDs should be a list of length Batch
    assert len(ids) == 8, "ID list length mismatch"

    print("Data Loading verification successful.")


def demo_feature_extraction():
    print("\n=== Demo 2: Feature Extraction (Single Config) ===")
    # Demonstrate run_extraction for a specific backbone/view combination.
    # This creates cache files that CrossValidator will pick up later.

    backbone = "convnext"
    view = "warped"
    split = "train"

    print(f"Running extraction for: {backbone} | {view} | {split}")

    # Create debug loader
    loader = get_dataloader(
        csv_path=Config.TRAIN_METADATA_PATH,
        batch_size=16,
        view_mode=view,
        backbone_type=backbone,
        shuffle=False,  # Must be False for feature extraction to align with IDs
        debug=True,
    )

    # Run extraction
    # load_cached_data=False forces it to run (unless we want to test cache logic)
    # We set it to False to prove the extractor works.
    features, ids, meta, targets = run_extraction(
        dataloader=loader,
        backbone_key=backbone,
        split=split,
        view_mode=view,
        load_cached_data=False,
    )

    print(f"Extracted Features Shape: {features.shape}")

    # Assertions
    # Debug mode = 64 samples
    assert features.shape[0] == 64, f"Expected 64 samples, got {features.shape[0]}"
    # ConvNeXt Large output dim is 1536
    assert (
        features.shape[1] == 1536
    ), f"Expected 1536 dimensions for ConvNeXt, got {features.shape[1]}"

    print("Feature Extraction verification successful.")


def demo_full_pipeline():
    print("\n=== Demo 3: Full Cross-Validation Pipeline ===")
    print("Initializing CrossValidator with debug=True...")

    # CrossValidator(debug=True) will:
    # 1. Iterate over all backbones (clip, dinov2, convnext) and views (warped, preserved).
    # 2. Extract features for train/val/test splits (using debug subset).
    #    Note: It will reuse the 'convnext/warped/train' features generated in Demo 2.
    # 3. Train Level-0 experts (Ridge, SVR, ExtraTrees) using 5-Fold CV.
    # 4. Train Level-1 Meta-Learner.
    # 5. Generate submission.csv.

    validator = CrossValidator(debug=True)

    print("Executing pipeline...")
    cv_rmse, val_rmse = validator.run()

    print(f"\nPipeline Execution Completed.")
    print(f"Ensemble CV RMSE: {cv_rmse:.4f}")
    print(f"Hold-out Val RMSE: {val_rmse:.4f}")

    # Verify Submission
    sub_path = Config.SUBMISSION_PATH
    print(f"Verifying submission file at: {sub_path}")

    if not os.path.exists(sub_path):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(sub_path)
    print(f"Submission Dataframe Shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head().to_string())

    # In debug mode, the test set is also truncated to 64 samples.
    assert (
        len(df_sub) == 64
    ), f"Expected 64 predictions in debug mode, found {len(df_sub)}"

    # Check for NaN values
    if df_sub["Pawpularity"].isnull().any():
        raise ValueError("Submission contains NaN values.")

    print("Full Pipeline verification successful.")


if __name__ == "__main__":
    # Configuration for the run
    seed_everything(Config.SEED)

    # Suppress verbose warnings from libraries
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        # Step 1: Verify Data Loading
        demo_data_loading()

        # Step 2: Verify Feature Extraction (and prime cache)
        demo_feature_extraction()

        # Step 3: Run Full Pipeline
        demo_full_pipeline()

        print("\nAll demonstrations passed.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        # Re-raise to ensure non-zero exit code if wrapper checks it
        raise e
