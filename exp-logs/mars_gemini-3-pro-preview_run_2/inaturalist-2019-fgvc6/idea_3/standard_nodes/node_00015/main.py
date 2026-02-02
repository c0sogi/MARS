import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

# Import from provided library
from library.config import CFG
from library.utils import seed_everything, save_checkpoint, AverageMeter, accuracy
from library.dataset import get_loaders
from library.model import build_model
from library.engine import train_one_epoch, validate


def run_analysis_and_submission(
    model, val_loader, test_loader, device, best_metric, threshold
):
    """
    Performs failure analysis and generates submission if criteria are met.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    # --- Failure Analysis on Validation Set ---
    val_metadata = pd.read_csv(CFG.val_metadata_path)
    train_metadata = pd.read_csv(CFG.train_metadata_path)

    # Calculate class frequencies from training data
    class_counts = train_metadata["category_id"].value_counts().to_dict()
    val_metadata["class_freq"] = val_metadata["category_id"].map(class_counts).fillna(0)

    # Get image dimensions (using a heuristic or reading if available,
    # but since we don't want to read all images, we'll skip dimension correlation
    # if not easily available, or just infer from a few if needed.
    # Actually, the prompt asks to correlate with input features.
    # We can assume the metadata might not have width/height columns unless generated.
    # The generated metadata in the prompt description only has id, file_name, category.
    # We will focus on class frequency which is available.)

    all_preds = []
    all_targets = []
    all_image_ids = []

    # Custom validation loop to collect predictions
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Inference
            with torch.cuda.amp.autocast():
                outputs = model(images)

            # Get predictions
            _, preds = outputs.topk(1, 1, True, True)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

            # We assume loader order matches metadata order because shuffle=False for val

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate errors (1 for incorrect, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    # Add to metadata
    # Ensure length matches
    if len(errors) == len(val_metadata):
        val_metadata["error"] = errors

        # Correlation with Class Frequency
        corr, p_val = pearsonr(val_metadata["error"], val_metadata["class_freq"])
        print(
            f"Correlation between Error and Class Frequency: {corr:.4f} (p={p_val:.4f})"
        )
    else:
        print("Warning: Mismatch in validation set size for analysis.")

    # --- Submission ---
    print(f"\nComparing Best Metric {best_metric:.6f} with Threshold {threshold:.6f}")
    if best_metric < threshold:
        print("Metric threshold met. Generating submission...")

        submission_preds = []
        submission_ids = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)

                # Get top 5
                _, top5_preds = outputs.topk(5, 1, True, True)

                submission_preds.append(top5_preds.cpu().numpy())
                submission_ids.append(ids.numpy())

        submission_preds = np.concatenate(submission_preds)
        submission_ids = np.concatenate(submission_ids)

        # Format for CSV
        # id,predicted
        # 123,1 2 3 4 5

        df_sub = pd.DataFrame(
            {
                "id": submission_ids,
                "predicted": [" ".join(map(str, row)) for row in submission_preds],
            }
        )

        sub_path = os.path.join(CFG.submission_dir, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print("Metric threshold NOT met. Skipping submission.")


def main():
    # 1. Configuration Override for Fast Baseline
    # Limit epochs to ensure completion within 2 hours
    CFG.epochs = 5
    print(f"Modified CFG.epochs to {CFG.epochs} for fast baseline execution.")

    # 2. Setup
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    # 3. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders()

    # 4. Model Construction
    print(f"Building model: {CFG.model_name}")
    model = build_model()
    model.to(device)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    # 6. Training Loop
    best_error = float("inf")

    print("Starting training...")
    for epoch in range(CFG.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Metric: Top-1 Error
        val_error = 1.0 - (val_acc / 100.0)

        # Save Checkpoint
        is_best = val_error < best_error
        if is_best:
            best_error = val_error

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_error": best_error,
            },
            is_best,
        )

        print(
            f"Epoch {epoch} Summary: Val Error: {val_error:.4f} (Best: {best_error:.4f})"
        )

    # 7. Final Evaluation & Analysis
    print("\nLoading best model for final evaluation...")
    checkpoint_path = os.path.join(CFG.output_dir, "model_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # Re-calculate metric on full validation set to be precise and print required format
    print("Running final validation...")
    final_loss, final_acc = validate(model, val_loader, device)
    final_metric = 1.0 - (final_acc / 100.0)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 8. Analysis and Submission
    # Threshold from prompt: 0.2647424892703862
    THRESHOLD = 0.2647424892703862
    run_analysis_and_submission(
        model, val_loader, test_loader, device, final_metric, THRESHOLD
    )


if __name__ == "__main__":
    main()
