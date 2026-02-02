import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_metadata, get_dataloader, PetDataset
from library.feature_extraction import FeatureExtractor
from library.stacking_engine import StackingEngine


def run_pipeline_demonstration():
    print("============================================================")
    print("  Pet Pawpularity: End-to-End Pipeline Demonstration")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config attributes for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 30  # Small subset for speed
    Config.N_FOLDS = 2  # Minimal folds for Cross-Validation
    Config.BATCH_SIZE = 4  # Small batch size

    # Reduce complexity of Level-0 and Level-1 models
    Config.ET_N_ESTIMATORS = 10  # Fewer trees for ExtraTrees
    Config.RIDGE_ALPHAS = [1.0, 10.0]  # Fewer alphas for RidgeCV
    Config.META_N_ITER = 10  # Fewer iterations for Meta-Learner

    # Limit to a single backbone to avoid downloading/running multiple large models
    # This ensures the demo completes within the time limit
    Config.BACKBONES = {
        "convnext": "facebook/convnext-large-224-22k-1k",
    }

    # Ensure working directory is clean for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(
        f"    Config updated: DEBUG={Config.DEBUG}, Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )
    print(f"    Backbone selected: {list(Config.BACKBONES.keys())[0]}")

    # ------------------------------------------------------------------
    # 2. Data Loading Verification
    # ------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load metadata subset
    df_train = load_metadata(mode="train", debug=True)
    print(f"    Loaded train metadata shape: {df_train.shape}")

    # Verify Dataset logic
    dataset = PetDataset(df_train, backbone_name="convnext")
    sample = dataset[0]

    print(f"    Sample ID: {sample['id']}")
    print(f"    Global View Shape: {sample['global_view'].shape}")
    print(f"    Local View Shape:  {sample['local_view'].shape}")
    print(f"    Meta Features:     {sample['meta'].shape}")

    # Assertions to ensure data integrity
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), "Dataset size does not match debug config"
    assert sample["global_view"].shape == (
        3,
        224,
        224,
    ), "Global view tensor shape mismatch"
    assert sample["local_view"].shape == (
        3,
        224,
        224,
    ), "Local view tensor shape mismatch"
    assert sample["meta"].shape == (12,), "Metadata feature vector shape mismatch"

    # Verify DataLoader
    loader = get_dataloader(
        mode="train", backbone_name="convnext", batch_size=Config.BATCH_SIZE
    )
    batch = next(iter(loader))
    assert (
        batch["global_view"].shape[0] == Config.BATCH_SIZE
    ), "DataLoader batch size mismatch"
    print("    Data Loading checks passed.")

    # ------------------------------------------------------------------
    # 3. Feature Extraction
    # ------------------------------------------------------------------
    print("\n[3] Running Feature Extraction...")
    print("    (This extracts features for the subset using the ConvNeXt backbone)")

    extractor = FeatureExtractor()
    # Run extraction (load_cached_data=False forces computation)
    extractor.run(load_cached_data=False)

    # Verify that feature files were created in the working directory
    print("    Verifying generated feature files...")
    modes = ["train_all", "test"]
    required_files = []

    for mode in modes:
        # Check for backbone-specific files
        for backbone in Config.BACKBONES.keys():
            required_files.append(f"{mode}_{backbone}_global_features.npy")
            required_files.append(f"{mode}_{backbone}_local_features.npy")
        # Check for common files
        required_files.append(f"{mode}_ids.npy")
        required_files.append(f"{mode}_meta.npy")
        required_files.append(f"{mode}_targets.npy")

    for fname in required_files:
        fpath = os.path.join(Config.WORKING_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected feature file not found: {fpath}")

    print("    All feature extraction files verified.")

    # ------------------------------------------------------------------
    # 4. Stacking Engine Execution
    # ------------------------------------------------------------------
    print("\n[4] Running Stacking Engine...")
    print("    (Training Level-0 Experts -> Level-1 Meta Learner -> Submission)")

    engine = StackingEngine()
    # Run the engine (load_cached_data=False forces retraining of experts)
    engine.run(load_cached_data=False)

    # ------------------------------------------------------------------
    # 5. Submission Validation
    # ------------------------------------------------------------------
    print("\n[5] Validating Submission...")

    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not created at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission File: {submission_path}")
    print(f"    Shape: {sub_df.shape}")
    print(sub_df.head())

    # Validate submission content
    # Note: In DEBUG mode, the test set is also subsampled
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count {len(sub_df)} != Debug Sample Size {Config.DEBUG_SAMPLE_SIZE}"

    # Check for valid probability range
    preds = sub_df["Pawpularity"].values
    min_pred, max_pred = preds.min(), preds.max()
    print(f"    Predictions - Min: {min_pred:.4f}, Max: {max_pred:.4f}")

    assert min_pred >= 1.0, "Found predictions below 1.0"
    assert max_pred <= 100.0, "Found predictions above 100.0"
    assert not np.isnan(preds).any(), "Found NaN values in predictions"

    print("\n============================================================")
    print("  Pipeline Demonstration Completed Successfully!")
    print("============================================================")


if __name__ == "__main__":
    run_pipeline_demonstration()
