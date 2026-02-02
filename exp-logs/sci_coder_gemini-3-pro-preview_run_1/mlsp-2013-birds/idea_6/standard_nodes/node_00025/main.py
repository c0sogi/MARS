import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, Logger, save_checkpoint, calculate_roc_auc
from library.dataset import get_data_loaders
from library.model import BirdResNet
from library.engine import train_one_epoch, validate, inference


def main():
    # --- 1. Setup & Configuration ---
    set_seed(Config.SEED)

    # Initialize Logger
    logger = Logger(os.path.join(Config.WORKING_DIR, "training_log.txt"))
    logger.log("Starting ResNet-34 Supervised Pipeline (Replicated Channels)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    # --- 2. Data Loading ---
    logger.log("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = get_data_loaders(Config)

    # --- 3. Training ---
    logger.log("\n--- Training ---")

    model = BirdResNet(num_classes=Config.NUM_CLASSES, pretrained=True).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_auc = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, Config)
        val_loss, val_auc = validate(model, val_loader, device)
        scheduler.step()

        logger.log_metrics(epoch, train_loss, val_loss, val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, Config.BEST_MODEL_PATH)
            logger.log(f"  -> New Best Model Saved! AUC: {val_auc:.4f}")

    logger.log(f"Best Validation AUC: {best_auc:.4f}")

    # --- 4. Final Evaluation & Failure Analysis ---
    logger.log("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Final Validation
    _, final_val_auc = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    model.eval()
    val_errors = []
    val_img_means = []
    val_img_stds = []
    val_label_counts = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Calculate Mean Absolute Error per sample
            mae = torch.abs(probs - labels).mean(dim=1).cpu().numpy()
            val_errors.extend(mae)

            # Image stats (Channel 0 is intensity)
            img_mean = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
            img_std = images[:, 0, :, :].std(dim=(1, 2)).cpu().numpy()

            val_img_means.extend(img_mean)
            val_img_stds.extend(img_std)

            # Label counts
            l_counts = labels.sum(dim=1).cpu().numpy()
            val_label_counts.extend(l_counts)

    val_errors = np.array(val_errors)
    val_img_means = np.array(val_img_means)
    val_img_stds = np.array(val_img_stds)
    val_label_counts = np.array(val_label_counts)

    # Correlations
    corr_mean, _ = pearsonr(val_errors, val_img_means)
    corr_std, _ = pearsonr(val_errors, val_img_stds)
    corr_count, _ = pearsonr(val_errors, val_label_counts)

    print("\nFailure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"  Signal Intensity (Mean): {corr_mean:.4f}")
    print(f"  Signal Contrast (Std):   {corr_std:.4f}")
    print(f"  Label Complexity (Count): {corr_count:.4f}")

    # --- 5. Submission ---
    threshold = 0.9255537489325414
    if final_val_auc > threshold:
        logger.log(
            f"\nValidation metric {final_val_auc} > {threshold}. Generating submission..."
        )

        # Inference on Test Set
        test_ids, test_probs = inference(model, test_loader, device)

        # Format Submission
        submission_rows = []
        for i in range(len(test_ids)):
            rec_id = int(test_ids[i])
            probs = test_probs[i]
            for species_idx in range(len(probs)):
                row_id = rec_id * 100 + species_idx
                prob = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.log(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.log(
            f"\nValidation metric {final_val_auc} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
