import os
import sys
import importlib
import library.config
import library.utils
import library.data
import library.model
import library.train_eval

# Reload modules to ensure config and model changes are picked up (Cite Debug Lesson 4)
importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train_eval)

import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device, setup_logger, save_checkpoint
from library.data import get_dataloaders, get_test_dataloader
from library.model import CervicalFractureModel
from library.train_eval import train_one_epoch, validate, WeightedMultilabelLoss


def analyze_failures(model, val_loader, criterion, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and input slice count.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    study_losses = []
    slice_counts = []

    # Access the dataset to get path maps for slice counting
    dataset = val_loader.dataset

    # We need to manually compute loss per sample, so we replicate the loss logic
    # without the final mean reduction
    bce_func = torch.nn.BCEWithLogitsLoss(reduction="none")
    weights = torch.tensor(Config.LOSS_WEIGHTS, device=device).view(1, -1)

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward
            logits = model(images)

            # Calculate weighted loss per sample (Batch, 8)
            raw_loss = bce_func(logits, targets)
            weighted_loss = raw_loss * weights

            # Mean over classes to get a single scalar error metric per study
            # Shape: (Batch,)
            sample_losses = weighted_loss.mean(dim=1).cpu().numpy()

            study_losses.extend(sample_losses)

            # Get slice counts for this batch
            # We need to map back to StudyInstanceUIDs to get slice counts
            # Since shuffle=False for val_loader, we can index the metadata directly
            start_idx = i * val_loader.batch_size
            end_idx = start_idx + images.size(0)

            batch_uids = dataset.metadata.iloc[start_idx:end_idx][
                "StudyInstanceUID"
            ].values

            for uid in batch_uids:
                # dataset.path_map is {uid: [list of paths]}
                count = len(dataset.path_map.get(uid, []))
                slice_counts.append(count)

    # Calculate Correlation
    if len(study_losses) > 1:
        correlation = np.corrcoef(study_losses, slice_counts)[0, 1]
        print(f"Correlation (Error Magnitude vs Slice Count): {correlation:.10f}")
    else:
        print("Insufficient validation samples for correlation analysis.")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\n=== Generating Submission ===")

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # Load Test Data
    test_loader = get_test_dataloader(load_cached_data=True)
    model.eval()

    results = []

    # Config.TARGET_COLS order: ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    target_cols = Config.TARGET_COLS

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Test loader returns images only, or images and dummy targets?
            # data.py __getitem__ returns (seq, target) if labels exist, else just seq?
            # Looking at data.py: if self.labels is not None...
            # Test metadata usually doesn't have targets.
            # However, data.py checks "if 'patient_overall' in self.metadata.columns".
            # test_metadata.csv created in metadata step only has StudyInstanceUID and image_path.
            # So __getitem__ returns just sequence_tensor.

            # Handle both cases just to be safe
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # Shape (8,)

            # Get StudyUID from metadata
            # Batch size is 1 for test loader
            uid = test_loader.dataset.metadata.iloc[i]["StudyInstanceUID"]

            # Format rows
            for col_idx, col_name in enumerate(target_cols):
                row_id = f"{uid}_{col_name}"
                prob = probs[col_idx]
                results.append({"row_id": row_id, "fractured": prob})

    # Save
    sub_df = pd.DataFrame(results)
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with {len(sub_df)} rows.")


def main():
    # --- 1. Fast Baseline Configuration ---
    Config.EPOCHS = 5  # Reduced from 10 to ensure < 2h runtime
    Config.BATCH_SIZE = 2  # Kept small for memory safety

    # Setup
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger()
    device = get_device()
    seed_everything(Config.SEED)

    logger.info("Initializing Fast Baseline Run...")

    # --- 2. Data Loading ---
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # --- 3. Model Initialization ---
    model = CervicalFractureModel().to(device)

    # --- 4. Training Setup ---
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)
    criterion = WeightedMultilabelLoss(Config.LOSS_WEIGHTS, device)
    scaler = torch.cuda.amp.GradScaler()

    # --- 5. Training Loop ---
    best_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, scaler
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, logger)

        # Scheduler
        scheduler.step()

        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, scheduler, filename="best_model.pth"
            )
            logger.info(f"New best model saved with loss: {best_loss:.6f}")

    # --- 6. Final Evaluation ---
    # Load best model
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(Config.OUTPUT_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Calculate Final Metric
    final_val_loss = validate(model, val_loader, criterion, device, logger)
    print(f"Final Validation Metric: {final_val_loss}")

    # --- 7. Failure Analysis ---
    analyze_failures(model, val_loader, criterion, device)

    # --- 8. Submission ---
    THRESHOLD = 0.1241588886
    if final_val_loss < THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Validation metric {final_val_loss} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
