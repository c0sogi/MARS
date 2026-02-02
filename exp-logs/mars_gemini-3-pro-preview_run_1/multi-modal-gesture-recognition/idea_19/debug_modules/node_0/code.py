import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

import library.config as config
from library.utils import set_seed, decode_predictions, levenshtein_distance
from library.data_loader import GestureDataset, collate_fn
from library.model import GCINet
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Configuration Override
    # Create a specific working directory for this demo to avoid conflicts
    demo_work_dir = "./working/demo_execution"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir, exist_ok=True)

    # Override paths in config to point to our demo directory
    config.WORKING_DIR = demo_work_dir
    config.CHECKPOINT_DIR = os.path.join(demo_work_dir, "checkpoints")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Override Hyperparameters for speed
    config.HYPERPARAMS["num_epochs"] = 2
    config.HYPERPARAMS["batch_size"] = 4
    config.HYPERPARAMS["hidden_dim"] = 64  # Smaller model
    config.HYPERPARAMS["early_stopping_patience"] = 2
    config.HYPERPARAMS["n_mfcc"] = 13

    # Set seed for reproducibility
    set_seed(config.HYPERPARAMS["seed"])

    print("Configuration updated for fast execution.")

    # 2. Create Subset Metadata (Data Preparation)
    # We use the existing metadata but slice it to create a tiny dataset for the demo.
    # This speeds up the 'stats computation' in GestureDataset significantly.

    subset_size = 10
    meta_dir = os.path.join(demo_work_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # Helper to create subset
    def create_subset(src_path, dst_name):
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Filter for valid samples (files must exist)
            # The provided loader filters by notna(), we do the same here plus existence check
            valid_indices = []
            for idx, row in df.iterrows():
                if pd.notna(row["data_path"]) and os.path.exists(
                    os.path.join(config.INPUT_DIR, row["data_path"])
                ):
                    valid_indices.append(idx)
                if len(valid_indices) >= subset_size:
                    break

            subset_df = df.loc[valid_indices]
            dst_path = os.path.join(meta_dir, dst_name)
            subset_df.to_csv(dst_path, index=False)
            return dst_path
        return None

    print("Creating metadata subsets...")
    new_train_path = create_subset(config.TRAIN_METADATA_PATH, "train.csv")
    new_val_path = create_subset(config.VAL_METADATA_PATH, "val.csv")
    new_test_path = create_subset(config.TEST_METADATA_PATH, "test.csv")

    # Update config paths to point to these new subsets
    config.TRAIN_METADATA_PATH = new_train_path
    config.VAL_METADATA_PATH = new_val_path
    config.TEST_METADATA_PATH = new_test_path

    # 3. Verify Utils
    print("\n=== Verifying Utils ===")
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical sequences should be 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert (
        dist_diff == 1
    ), f"Levenshtein distance for deletion should be 1, got {dist_diff}"

    # Test Decode
    # 0 is background. Sequence: 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0
    # Min len 5. Both 1 and 2 should be detected.
    raw_preds = np.array([0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0])
    decoded = decode_predictions(
        raw_preds, background_id=0, min_len=5, median_filter_size=1
    )
    assert decoded == [1, 2], f"Decoding failed. Expected [1, 2], got {decoded}"
    print("Utils verification passed.")

    # 4. Dataset and DataLoader
    print("\n=== Initializing Dataset and DataLoader ===")
    # Initialize Train Dataset (this will trigger stats computation on the subset)
    train_dataset = GestureDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        split="train",
        load_cached_data=False,  # Force recompute for demo
    )

    # Initialize Val Dataset
    val_dataset = GestureDataset(
        metadata_path=config.VAL_METADATA_PATH, split="val", load_cached_data=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.HYPERPARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.HYPERPARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    skeletons, audios, lengths, labels = batch

    print(
        f"Batch Shapes -> Skeletons: {skeletons.shape}, Audios: {audios.shape}, Labels: {labels.shape}"
    )

    # Assertions
    # Skeletons: (Batch, Time, 60)
    assert skeletons.dim() == 3 and skeletons.shape[2] == 60, "Skeleton shape mismatch"
    # Audios: (Batch, Time, 13)
    assert audios.dim() == 3 and audios.shape[2] == 13, "Audio shape mismatch"
    # Labels: (Batch, Time)
    assert labels.dim() == 2, "Labels shape mismatch"

    print("Data loading verification passed.")

    # 5. Model Initialization and Forward Pass
    print("\n=== Initializing GCINet Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCINet().to(device)

    # Move batch to device
    skeletons = skeletons.to(device)
    audios = audios.to(device)
    lengths = lengths.to(device)

    # Forward pass
    logits = model(skeletons, audios, lengths)
    print(f"Logits Shape: {logits.shape}")

    # Assertions
    # Logits: (Batch, Time, NumClasses)
    # Total classes = 20 gestures + 1 background = 21
    assert logits.shape[0] == skeletons.shape[0], "Batch size mismatch in output"
    assert logits.shape[1] == skeletons.shape[1], "Time dimension mismatch in output"
    assert (
        logits.shape[2] == config.TOTAL_CLASSES
    ), f"Class dimension mismatch. Expected {config.TOTAL_CLASSES}, got {logits.shape[2]}"

    print("Model forward pass verification passed.")

    # 6. Training Loop Demo
    print("\n=== Starting Training Loop Demo ===")
    trainer = Trainer(train_loader, val_loader, device=device)
    trainer.fit()

    # Check if checkpoint was created
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint successfully created at {best_model_path}")
    else:
        print(
            "Warning: No checkpoint created (validation might have been poor or empty)."
        )

    # 7. Inference and Submission Generation
    print("\n=== Generating Submission (Mock) ===")
    # Load Test Dataset
    test_dataset = GestureDataset(
        metadata_path=config.TEST_METADATA_PATH, split="test", load_cached_data=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Sequential processing for submission
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    submission_lines = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Get data
            skeletons, audios, lengths, _ = batch
            skeletons = skeletons.to(device)
            audios = audios.to(device)
            lengths = lengths.to(device)

            # Predict
            logits = model(skeletons, audios, lengths)
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()[0]

            # Decode
            curr_len = lengths.cpu().numpy()[0]
            valid_preds = preds[:curr_len]

            pred_gestures = decode_predictions(
                valid_preds,
                background_id=config.BACKGROUND_CLASS_ID,
                min_len=config.HYPERPARAMS["min_gesture_length"],
                median_filter_size=config.HYPERPARAMS["median_filter_size"],
            )

            # Format: SessionID,Label1,Label2...
            # We need the SessionID. It's in the metadata.
            sample_id = test_dataset.metadata.iloc[i]["sample_id"]

            line = f"{sample_id}," + ",".join(map(str, pred_gestures))
            submission_lines.append(line)

    # Save Submission
    submission_dir = os.path.join(demo_work_dir, "submission")
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    with open(submission_path, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission file generated at: {submission_path}")
    print("First 3 lines of submission:")
    for l in submission_lines[:3]:
        print(l)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
