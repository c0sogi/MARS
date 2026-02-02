import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, do_kaggle_metric
from library.dataset import get_loaders
from library.model import DeepResUNet
from library.loss import BCEDiceLoss
from library.train import Trainer
from library.evaluate import Evaluator


def main():
    print("=== Salt Segmentation Pipeline Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("[1] Configuring environment for rapid execution...")

    # Reduce training duration
    Config.EPOCHS_PER_CYCLE = 1
    Config.CYCLES = 2
    Config.TOTAL_EPOCHS = 2
    Config.CYCLE_1_END_EPOCH = 1

    # Reduce computational load
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Clean and setup working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()

    # Set seed
    seed_everything(Config.SEED)
    print(f"    Total Epochs: {Config.TOTAL_EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")
    # Load debug subset (100 samples)
    train_loader, val_loader = get_loaders(debug=True, load_cached_data=False)

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Assertions
    print(
        f"    Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Image batch shape mismatch"
    assert masks.shape == (Config.BATCH_SIZE, 1, 128, 128), "Mask batch shape mismatch"
    assert depths.shape == (Config.BATCH_SIZE,), "Depth batch shape mismatch"
    assert len(ids) == Config.BATCH_SIZE, "ID list length mismatch"
    print("    Dataset loaded and verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")
    device = Config.DEVICE
    model = DeepResUNet().to(device)

    # Move data to device
    img_tensor = images.to(device)
    mask_tensor = masks.to(device)
    depth_tensor = depths.to(device)

    # Forward pass (Train mode -> Deep Supervision returns list)
    model.train()
    outputs = model(img_tensor, depth_tensor)

    assert isinstance(outputs, list), "Model output should be a list in training mode"
    assert len(outputs) == 3, "Model should return 3 outputs (Deep Supervision)"
    print(f"    Model output shapes: {[o.shape for o in outputs]}")

    # Calculate Loss
    criterion = BCEDiceLoss()
    loss = criterion(outputs[0], mask_tensor)

    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # Calculate Metric
    metric = do_kaggle_metric(outputs[0], mask_tensor, threshold=0.5)
    print(f"    Calculated mAP (on random weights): {metric:.4f}")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Trainer)...")
    trainer = Trainer(debug=True)
    trainer.train()

    # -------------------------------------------------------------------------
    # 5. Checkpoint Preparation for Evaluation
    # -------------------------------------------------------------------------
    print("\n[5] Preparing Checkpoints for Ensemble...")
    # The Evaluator expects 'best_cycle_2.pth', 'best_cycle_3.pth', 'best_cycle_4.pth'.
    # Our short training run might not produce all of them, or might not improve mAP to save.
    # We manually save the current model state to these paths to ensure Evaluator runs.

    checkpoint_dir = Config.CHECKPOINT_DIR
    required_checkpoints = ["best_cycle_2.pth", "best_cycle_3.pth", "best_cycle_4.pth"]
    current_state = model.state_dict()

    for cp_name in required_checkpoints:
        cp_path = os.path.join(checkpoint_dir, cp_name)
        if not os.path.exists(cp_path):
            print(f"    Creating mock checkpoint: {cp_name}")
            torch.save(current_state, cp_path)
        else:
            print(f"    Checkpoint exists: {cp_name}")

    # -------------------------------------------------------------------------
    # 6. Evaluation & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Executing Evaluation and Submission (Evaluator)...")
    evaluator = Evaluator(debug=True)

    # Run Gated Ensemble
    # This validates the checkpoints, ensembles the good ones, and predicts on the test set.
    evaluator.gated_ensemble()

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"    Submission file created: {submission_path}")
        print(f"    Rows: {len(df_sub)}")
        print("    Head:")
        print(df_sub.head())

        # Validation
        assert (
            len(df_sub) == 1000
        ), "Submission should contain 1000 rows (Test Set size)"
        assert (
            "id" in df_sub.columns and "rle_mask" in df_sub.columns
        ), "Invalid submission columns"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
