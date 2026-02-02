import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import provided libraries
from library.config import (
    seed_everything,
    LEARNING_RATE,
    WEIGHT_DECAY,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import GestureGRU
from library.trainer import Trainer
from library.inference import Predictor
from library.utils import (
    calculate_levenshtein_distance,
    decode_predictions,
    rle_collapse,
)


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast Baseline Config
    # We use a moderate batch size and limited epochs to ensure the script completes quickly
    BATCH_SIZE = 16
    EPOCHS = 20

    # 2. Data Loading
    # We use load_cached_data=True to use existing processed data
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH, load_cached_data=True, mode="train"
    )
    val_dataset = GestureDataset(VAL_METADATA_PATH, load_cached_data=True, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model & Optimizer
    model = GestureGRU()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, optimizer, device)
    trainer.fit(epochs=EPOCHS)

    # 5. Final Validation & Failure Analysis
    # Load best model saved by the trainer
    if os.path.exists(trainer.checkpoint_path):
        model.load_state_dict(torch.load(trainer.checkpoint_path, map_location=device))

    model.eval()

    val_errors = []
    val_seq_lengths = []
    val_num_gestures = []

    total_dist = 0
    total_gestures_count = 0

    # Manual validation loop to gather detailed stats for failure analysis
    with torch.no_grad():
        for features, targets, lengths, ids in val_loader:
            features = features.to(device)
            lengths = lengths.to(device)

            logits = model(features, lengths)

            # Move to CPU for processing
            logits_np = logits.cpu().numpy()
            targets_np = targets.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            for i in range(len(ids)):
                length = lengths_np[i]
                valid_logits = logits_np[i, :length, :]
                valid_targets = targets_np[i, :length]

                # Decode predictions and ground truth
                pred_seq = decode_predictions(valid_logits)
                target_seq = rle_collapse(
                    valid_targets, remove_background=True, background_class=0
                )

                # Compute Metric for this sample
                dist = calculate_levenshtein_distance(target_seq, pred_seq)
                num_g = len(target_seq)

                # Store stats
                val_errors.append(dist)
                val_seq_lengths.append(length)
                val_num_gestures.append(num_g)

                total_dist += dist
                total_gestures_count += num_g

    # Compute Global Metric
    final_metric = (
        total_dist / total_gestures_count if total_gestures_count > 0 else 0.0
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("Performing failure analysis...")
    df_analysis = pd.DataFrame(
        {"error": val_errors, "length": val_seq_lengths, "complexity": val_num_gestures}
    )

    # Calculate correlations
    corr_length = df_analysis["error"].corr(df_analysis["length"])
    corr_complexity = df_analysis["error"].corr(df_analysis["complexity"])

    print(f"Correlation (Error vs Sequence Length): {corr_length}")
    print(f"Correlation (Error vs Number of Gestures): {corr_complexity}")

    # 6. Submission
    # Generate predictions only if validation metric is improved (lower than 0.2891)
    if final_metric < 0.2891:
        print(
            f"Validation metric {final_metric:.4f} < 0.2891. Generating submission..."
        )
        predictor = Predictor(model_path=trainer.checkpoint_path, device=device)
        predictor.run_inference(
            test_metadata_path=TEST_METADATA_PATH,
            output_filename="submission.csv",
            batch_size=BATCH_SIZE,
            load_cached_data=True,
        )
        print("Submission generated successfully.")
    else:
        print(
            f"Validation metric {final_metric:.4f} >= 0.2891. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
