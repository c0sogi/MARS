import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings
from library import config, utils, data_loader, model, loss, engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting HG-GCRCN Demo Script ===")

    # 1. Setup directories for the demo run
    # We use a specific subdirectory in 'working' to avoid conflicts and ensure a clean state
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    demo_meta_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_meta_dir)

    # 2. Create subset metadata for fast execution
    # We read the original metadata and save a tiny subset (top N rows)
    # This forces the data loader to only process these few samples.
    print("Creating metadata subsets for rapid testing...")
    subsets = {"train": 8, "val": 4, "test": 4}

    for split, count in subsets.items():
        original_csv = os.path.join("./metadata", f"{split}.csv")
        if not os.path.exists(original_csv):
            raise FileNotFoundError(f"Metadata file {original_csv} not found.")

        df = pd.read_csv(original_csv)
        # Take the first 'count' samples
        subset_df = df.head(count)

        target_csv = os.path.join(demo_meta_dir, f"{split}.csv")
        subset_df.to_csv(target_csv, index=False)
        print(
            f"  - Created {split} subset with {len(subset_df)} samples at {target_csv}"
        )

    # 3. Override Configuration Parameters
    # We modify the global config module variables to point to our demo directories
    # and reduce model complexity/training time.
    print("Configuring hyperparameters for demo...")

    # Path Overrides
    config.METADATA_DIR = demo_meta_dir
    config.WORKING_DIR = os.path.join(demo_dir, "cache")
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Hyperparameter Overrides
    config.HYPERPARAMS.update(
        {
            "batch_size": 2,  # Small batch size
            "num_epochs": 2,  # Only 2 epochs
            "hidden_dim": 32,  # Reduced hidden dimension
            "lstm_layers": 1,  # Single LSTM layer
            "tcn_layers": 2,  # Fewer TCN layers
            "tcn_dilations": [1, 2],  # Reduced dilation depth
            "dropout": 0.0,  # No dropout for tiny data stability
            "audio_n_mfcc": 13,
            "class_weights": [0.1] + [1.0] * 20,  # Keep standard weights
        }
    )

    # Set Random Seed for Reproducibility
    utils.set_seed(42)

    # 4. Validate Data Loader
    print("\n[Validation] Data Loader...")
    # num_workers=0 ensures main process loading, avoiding multiprocessing overhead/issues in demo
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.HYPERPARAMS["batch_size"], num_workers=0
    )

    # Fetch a single batch to verify structure
    try:
        batch = next(iter(train_loader))
        skeleton = batch["skeleton"]
        audio = batch["audio"]
        targets = batch["targets"]

        print(f"  - Batch Skeleton Shape: {skeleton.shape} (Expected: B, T, 12, 3)")
        print(f"  - Batch Audio Shape: {audio.shape} (Expected: B, T, 13)")

        # Assertions
        assert (
            skeleton.dim() == 4 and skeleton.shape[2] == 12 and skeleton.shape[3] == 3
        ), "Skeleton tensor shape mismatch."
        assert audio.dim() == 3 and audio.shape[2] == 13, "Audio tensor shape mismatch."
        assert (
            "cls" in targets and "mask" in targets
        ), "Targets dictionary missing keys."

    except StopIteration:
        raise RuntimeError(
            "Data loader returned no batches. Check metadata subset creation."
        )

    # 5. Validate Model Architecture
    print("\n[Validation] Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Using device: {device}")

    net = model.HGGCRCN().to(device)

    # Move batch to device
    skel_dev = skeleton.to(device)
    aud_dev = audio.to(device)
    mask_dev = targets["mask"].to(device)

    # Forward Pass
    stage_outputs = net(skel_dev, aud_dev, mask_dev)

    print(f"  - Number of stages output: {len(stage_outputs)} (Expected: 3)")
    assert len(stage_outputs) == 3, "Model must return outputs for 3 stages."

    # Check last stage output
    last_stage = stage_outputs[-1]
    cls_logits = last_stage["cls"]
    print(f"  - Output Class Logits Shape: {cls_logits.shape}")

    assert (
        cls_logits.shape[0] == config.HYPERPARAMS["batch_size"]
    ), "Batch dimension mismatch."
    assert cls_logits.shape[2] == config.NUM_CLASSES, "Class dimension mismatch."

    # 6. Validate Loss Function
    print("\n[Validation] Loss Function...")
    criterion = loss.HierarchicalLoss().to(device)

    # Prepare targets on device
    targets_dev = {k: v.to(device) for k, v in targets.items()}

    total_loss, metrics = criterion(stage_outputs, targets_dev)

    print(f"  - Total Loss: {total_loss.item():.4f}")
    print(f"  - Metrics: {list(metrics.keys())}")

    assert not torch.isnan(total_loss), "Loss is NaN."
    assert total_loss > 0, "Loss must be positive."

    # 7. Validate Utility Functions (Metric)
    print("\n[Validation] Levenshtein Metric...")
    # Example: Target [1, 2], Pred [1, 3] -> Distance 1 (substitution), Length 2 -> Score 0.5
    dummy_preds = [[1, 3]]
    dummy_targs = [[1, 2]]
    score = utils.compute_levenshtein_score(dummy_preds, dummy_targs)
    print(f"  - Calculated Score: {score} (Expected: 0.5)")
    assert abs(score - 0.5) < 1e-6, "Levenshtein score calculation incorrect."

    # 8. Run Full Training & Inference Pipeline
    print("\n[Execution] Running Training Loop...")
    # engine.run handles training, validation, checkpointing, and inference
    engine.run(
        train_loader, val_loader, test_loader, epochs=config.HYPERPARAMS["num_epochs"]
    )

    # 9. Verify Submission Output
    print("\n[Verification] Checking Submission File...")
    submission_file = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file, header=None)
        print(f"  - Submission file found at: {submission_file}")
        print(f"  - Rows generated: {len(df_sub)}")
        print(f"  - First few rows:\n{df_sub.head()}")

        # We expect 4 rows corresponding to the 4 test samples in our subset
        assert len(df_sub) == 4, f"Expected 4 predictions, found {len(df_sub)}."
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
