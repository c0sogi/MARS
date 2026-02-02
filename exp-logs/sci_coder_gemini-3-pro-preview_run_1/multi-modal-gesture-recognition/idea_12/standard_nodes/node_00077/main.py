import torch
import pandas as pd
import numpy as np
import os
import sys

# Import components from the provided library
from library.config import Config
from library.train import train_model, get_gt_sequences
from library.predict import generate_submission
from library.model import SCRNet
from library.data_loader import GestureDataset, collate_fn
from library.utils import set_seed, batch_decode, levenshtein_distance
from torch.utils.data import DataLoader


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    print("==================================================")
    print("Step 1: Training SCR-Net Baseline")
    print("==================================================")

    # Train the model
    # Using Config.NUM_EPOCHS (50) to ensure convergence. Cite {solution_lesson_node_00040}
    train_model(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=False)

    print("\n==================================================")
    print("Step 2: Validation & Failure Analysis")
    print("==================================================")

    device = torch.device(Config.DEVICE)

    # Load the best model checkpoint
    model = SCRNet().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Model checkpoint not found at {Config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Initialize Validation Loader
    val_dataset = GestureDataset(
        metadata_path=Config.VAL_METADATA_PATH, is_train=False, load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Containers for analysis
    val_preds = []
    val_targets = []
    val_frame_lengths = []
    val_target_lengths = []
    val_errors = []

    print("Running inference on validation set...")

    with torch.no_grad():
        for skeletons, audios, labels, lengths in val_loader:
            skeletons = skeletons.to(device)
            audios = audios.to(device)
            lengths = lengths.to(device)
            # labels are needed for GT generation

            # Forward pass
            logits = model(skeletons, audios, lengths)

            # Decode predictions
            batch_preds = batch_decode(logits, lengths)

            # Get Ground Truth sequences
            batch_targets = get_gt_sequences(labels, lengths)

            # Store results
            val_preds.extend(batch_preds)
            val_targets.extend(batch_targets)
            val_frame_lengths.extend(lengths.cpu().tolist())

            # Compute per-sample error for analysis
            for p, t in zip(batch_preds, batch_targets):
                dist = levenshtein_distance(p, t)
                val_errors.append(dist)
                val_target_lengths.append(len(t))

    # Compute Final Metric
    total_distance = sum(val_errors)
    total_len = sum(val_target_lengths)
    final_metric = total_distance / total_len if total_len > 0 else float("inf")

    # Print metric in required format
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    print("\nPerforming Failure Analysis...")

    df_analysis = pd.DataFrame(
        {
            "error": val_errors,
            "frame_length": val_frame_lengths,
            "target_seq_length": val_target_lengths,
        }
    )

    # Compute correlations
    corr_frame_len = df_analysis["error"].corr(df_analysis["frame_length"])
    corr_seq_len = df_analysis["error"].corr(df_analysis["target_seq_length"])

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  - Sequence Duration (Frames): {corr_frame_len:.10f}")
    print(f"  - Ground Truth Sequence Length (Gestures): {corr_seq_len:.10f}")

    print("\n==================================================")
    print("Step 3: Submission Generation")
    print("==================================================")

    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(
            f"Validation Metric ({final_metric:.6f}) is better than threshold ({THRESHOLD:.6f})."
        )
        generate_submission(
            checkpoint_path=Config.BEST_MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
            debug=False,
        )
    else:
        print(
            f"Validation Metric ({final_metric:.6f}) did not meet threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
