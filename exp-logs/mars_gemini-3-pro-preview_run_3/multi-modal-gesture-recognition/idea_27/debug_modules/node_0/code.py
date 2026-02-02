import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import random
import json

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.model import GHCKRN
from library.loss import CascadedLoss
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.utils import compute_levenshtein_score, levenshtein_distance


# ==========================================
# 1. Setup & Configuration Override
# ==========================================
def setup_environment():
    # Set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Override Config for Demo/Speed
    print("Setting up demo configuration...")

    # Use a temporary working directory for this run
    demo_dir = os.path.join("working", "demo_run_script")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.METADATA_DIR = os.path.join(demo_dir, "metadata")
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")

    # Create necessary subdirs
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.METADATA_DIR, exist_ok=True)

    # Reduce compute load
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"Working directory set to: {Config.WORKING_DIR}")


# ==========================================
# 2. Data Subsetting
# ==========================================
def create_subset_metadata():
    print("Creating subset metadata for rapid execution...")

    # Source paths
    src_meta_dir = "./metadata"
    files = ["train.csv", "val.csv", "test.csv"]

    # Number of samples to use for demo
    n_samples = 10

    for f in files:
        src_path = os.path.join(src_meta_dir, f)
        dst_path = os.path.join(Config.METADATA_DIR, f)

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Take a subset
            df_subset = df.head(n_samples).copy()
            df_subset.to_csv(dst_path, index=False)
            print(f"  Created {dst_path} with {len(df_subset)} samples.")
        else:
            # Fallback if source doesn't exist (should not happen based on problem description)
            print(f"  Warning: Source {src_path} not found. Creating empty.")
            pd.DataFrame(
                columns=[
                    "sample_id",
                    "rgb_path",
                    "depth_path",
                    "audio_path",
                    "user_path",
                    "data_path",
                    "labels",
                    "num_gestures",
                ]
            ).to_csv(dst_path, index=False)


# ==========================================
# 3. Component Validation
# ==========================================
def validate_components():
    print("Validating model and loss components...")

    # 3.1 Model Shape Check
    model = GHCKRN()
    model.to(Config.DEVICE)
    model.eval()

    # Batch=2, Time=64, Dim=193 (Config.INPUT_DIM)
    dummy_input = torch.randn(2, 64, Config.INPUT_DIM).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(dummy_input)

    # Check outputs
    assert isinstance(
        outputs, list
    ), "Model output should be a list (Cascaded architecture)"
    assert len(outputs) == 3, f"Expected 3 stages, got {len(outputs)}"

    # Check shape of last stage: (Batch, NumClasses, Time)
    last_out = outputs[-1]
    expected_shape = (2, Config.NUM_CLASSES, 64)
    assert (
        last_out.shape == expected_shape
    ), f"Output shape mismatch. Expected {expected_shape}, got {last_out.shape}"
    print("  Model forward pass successful.")

    # 3.2 Loss Check
    criterion = CascadedLoss().to(Config.DEVICE)

    # Create dummy targets (Batch, Time)
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (2, 64)).to(Config.DEVICE)

    loss = criterion(outputs, dummy_targets)
    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"  Loss computation successful. Value: {loss.item():.4f}")

    # 3.3 Utils Check
    hyp = [1, 2, 3]
    ref = [1, 2, 3]
    dist = levenshtein_distance(hyp, ref)
    assert dist == 0, "Levenshtein distance for identical sequences should be 0"

    hyp = [1, 2]
    ref = [1, 2, 3]
    dist = levenshtein_distance(hyp, ref)
    assert dist == 1, "Levenshtein distance for deletion should be 1"
    print("  Utils validation successful.")


# ==========================================
# 4. Pipeline Execution
# ==========================================
def run_pipeline():
    print("Starting full pipeline execution...")

    # 4.1 Data Loading
    # This will read from the subset metadata created in step 2
    dl_train, dl_val, dl_test = get_dataloaders(load_cached_data=False)

    print(f"  Train batches: {len(dl_train)}")
    print(f"  Val batches: {len(dl_val)}")
    print(f"  Test batches: {len(dl_test)}")

    # 4.2 Trainer Initialization
    model = GHCKRN()
    trainer = Trainer(model, dl_train, dl_val, dl_test)

    # 4.3 Training
    print("  Starting training loop...")
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Check if model file was created
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("  Model checkpoint found.")
    else:
        # If model wasn't saved (e.g. validation score didn't improve, or inf), save manually for inference test
        print(
            "  Model checkpoint not found (maybe validation score inf?). Saving manually for demo."
        )
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 4.4 Submission Generation
    print("  Generating submission...")
    trainer.generate_submission()

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
        print(f"  Submission file created with {len(lines)} lines.")

        # Check format of first line
        if len(lines) > 0:
            parts = lines[0].strip().split(",")
            # Format: SampleID, Label1, Label2...
            assert len(parts) >= 1, "Submission line too short"
            print(f"  Sample submission line: {lines[0].strip()}")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    try:
        setup_environment()
        create_subset_metadata()
        validate_components()
        run_pipeline()
        print("\n=== Demo Execution Completed Successfully ===")
    except Exception as e:
        print(f"\n!!! Demo Execution Failed: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
