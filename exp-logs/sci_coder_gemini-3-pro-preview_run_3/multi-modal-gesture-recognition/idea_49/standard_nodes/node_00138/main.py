import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.config import Config
from library.data_loader import GestureDataset
from library.model import SRDGN
from library.train import CascadedLoss, Trainer
from library.utils import (
    set_seed,
    setup_logger,
    compute_levenshtein,
    run_length_encoding,
)
from library.predict import predict


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("runfile", os.path.join(Config.WORKING_DIR, "run.log"))
    logger.info(f"Execution started on device: {device}")

    # 2. Data Loading
    # We use the full dataset but limit epochs for a fast baseline if needed.
    # The dataset is small (232 samples), so 30 epochs is very fast.
    train_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        mode="train",
        load_cached_data=True,
    )
    val_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), mode="val", load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # Validation batch size 1 for full sequence evaluation
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # 3. Model Initialization
    # Input dim: 180 (kinematics) + 13 (audio) = 193
    model = SRDGN(input_dim=193, num_classes=Config.NUM_CLASSES).to(device)

    # Loss with Class Weighting
    weights = torch.ones(Config.NUM_CLASSES).to(device)
    weights[0] = Config.BACKGROUND_WEIGHT
    criterion = CascadedLoss(weight=weights)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training
    logger.info("Starting training...")
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device
    )
    trainer.fit(epochs=Config.EPOCHS)

    # 5. Validation & Failure Analysis
    logger.info("Starting detailed validation and failure analysis...")

    # Load best model weights
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    total_dist = 0
    total_gestures = 0

    sample_errors = []
    sample_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            _, _, logits3 = model(features)

            # Decode predictions (Stage 3)
            preds = torch.argmax(logits3, dim=2).squeeze(0).cpu().numpy()
            targets = labels.squeeze(0).cpu().numpy()

            # Run-Length Encoding
            pred_seq = run_length_encoding(preds, min_length=Config.MIN_GESTURE_LENGTH)
            target_seq = run_length_encoding(
                targets, min_length=Config.MIN_GESTURE_LENGTH
            )

            # Compute Metric
            dist = compute_levenshtein(pred_seq, target_seq)
            total_dist += dist
            total_gestures += len(target_seq)

            # Collect data for failure analysis
            sample_errors.append(dist)
            sample_lengths.append(features.shape[1])  # Sequence length (frames)

    # Compute and print final metric
    final_metric = total_dist / total_gestures if total_gestures > 0 else float("inf")
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error and sequence length
    if len(sample_errors) > 1:
        correlation = np.corrcoef(sample_errors, sample_lengths)[0, 1]
        print(f"Correlation (Error vs Duration): {correlation:.4f}")

    # 6. Conditional Submission
    THRESHOLD = 0.16539050535987748

    if final_metric < THRESHOLD:
        logger.info(
            f"Metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        predict(load_cached_data=True)
    else:
        logger.info(
            f"Metric {final_metric} is NOT below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
