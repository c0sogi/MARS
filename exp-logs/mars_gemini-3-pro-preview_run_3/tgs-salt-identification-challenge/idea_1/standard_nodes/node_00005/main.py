import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library functions
from library.dataset import get_dataloaders
from library.model import LinkNetResNet34
from library.train import (
    BCEDiceLoss,
    train_epoch,
    valid_epoch,
    find_best_threshold,
    generate_submission,
)
from library.utils import do_kaggle_metric

# Configuration
SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = "./working"
SUBMISSION_DIR = "./submission"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def perform_failure_analysis(model, loader, device):
    """
    Analyzes model performance against metadata features (Depth, Salt Coverage).
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    # Load validation metadata to get features
    val_meta = pd.read_csv("./metadata/val_metadata.csv")
    # Create a map for quick lookup
    meta_map = val_meta.set_index("id")[["z", "coverage"]].to_dict("index")

    errors = []
    depths = []
    coverages = []

    # Crop parameters (128 -> 101)
    start_idx = 13
    end_idx = 13 + 101

    with torch.no_grad():
        for inputs, masks, ids in loader:
            inputs = inputs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # TTA for Failure Analysis
            outputs1, _ = model(inputs)
            inputs_flipped = torch.flip(inputs, [3])
            outputs2, _ = model(inputs_flipped)
            outputs2 = torch.flip(outputs2, [3])

            probs = (torch.sigmoid(outputs1) + torch.sigmoid(outputs2)) / 2.0

            # Crop to original size
            probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]
            masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

            preds_np = probs_cropped.cpu().numpy().squeeze(1)
            targets_np = masks_cropped.cpu().numpy().squeeze(1)

            # Calculate per-image Dice score to derive Error
            # Dice = 2*Intersection / (Sum + epsilon)
            intersection = (preds_np * targets_np).sum(axis=(1, 2))
            union = preds_np.sum(axis=(1, 2)) + targets_np.sum(axis=(1, 2))
            dice_scores = (2.0 * intersection + 1e-6) / (union + 1e-6)

            batch_errors = 1.0 - dice_scores

            for i, img_id in enumerate(ids):
                if img_id in meta_map:
                    errors.append(batch_errors[i])
                    depths.append(meta_map[img_id]["z"])
                    coverages.append(meta_map[img_id]["coverage"])

    # Calculate correlations
    if len(errors) > 0:
        corr_depth, _ = pearsonr(errors, depths)
        corr_cov, _ = pearsonr(errors, coverages)

        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

        if abs(corr_depth) > 0.1:
            print("Observation: Performance is sensitive to depth.")
        if abs(corr_cov) > 0.1:
            print(
                "Observation: Performance is sensitive to the amount of salt (class imbalance)."
            )
    else:
        print("Not enough data for failure analysis.")


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 3. Model
    print("Initializing model...")
    model = LinkNetResNet34(num_classes=1, pretrained=True)
    model = model.to(device)

    # 4. Optimization
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_metric = -1.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = valid_epoch(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # 6. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Compute final metric on validation set
    _, final_metric = valid_epoch(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Threshold Optimization
    print("\nOptimizing threshold...")
    best_threshold, best_score = find_best_threshold(model, val_loader, device)
    print(f"Optimal Threshold: {best_threshold:.4f} (Score: {best_score:.4f})")

    # 9. Submission
    if final_metric > 0.7957:
        print(f"Final metric {final_metric:.4f} > 0.7957. Generating submission...")
        generate_submission(model, test_loader, device, best_threshold, SUBMISSION_PATH)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Final metric {final_metric:.4f} did not exceed 0.7957. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
