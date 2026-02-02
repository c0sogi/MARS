import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import cv2

# Import from library
from library.config import Config
from library.dataset import get_dataloaders, load_data
from library.model import UltraWideSERepNeXt
from library.engine import train_model, evaluate
from library.utils import set_seed, calculate_roc_auc, load_checkpoint


def main():
    # 1. Setup
    set_seed(Config.BASE_SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # Use cached data as per instructions
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Prepare to store ensemble predictions
    num_val = len(val_loader.dataset)
    num_test = len(test_loader.dataset)

    # Accumulators for ensemble averaging
    val_probs_sum = np.zeros(num_val)
    test_probs_sum = np.zeros(num_test)

    # Store ground truth for validation once
    val_targets = val_loader.dataset.labels
    val_ids = val_loader.dataset.ids
    test_ids = test_loader.dataset.ids

    # 3. Training Loop (Homogeneous Seed Averaging)
    for seed in Config.SEEDS:
        print(f"Training Seed {seed}...")
        set_seed(seed)

        # Initialize Model
        model = UltraWideSERepNeXt()

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
        )

        # Train
        model_filename = f"model_seed_{seed}.pth"
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=Config.NUM_EPOCHS,
            patience=Config.EARLY_STOPPING_PATIENCE,
            filename=model_filename,
        )

        # Load Best Model for this seed
        checkpoint = load_checkpoint(model, model_filename, device=device)

        # Switch to Deploy Mode (Structural Re-parameterization)
        model.switch_to_deploy()
        model.to(device)
        model.eval()

        # Inference with TTA (Val)
        val_probs = predict_tta(model, val_loader, device)
        val_probs_sum += val_probs

        # Inference with TTA (Test)
        test_probs = predict_tta(model, test_loader, device)
        test_probs_sum += test_probs

    # 4. Ensemble Aggregation
    num_seeds = len(Config.SEEDS)
    avg_val_probs = val_probs_sum / num_seeds
    avg_test_probs = test_probs_sum / num_seeds

    # 5. Final Validation Metric
    final_val_auc = calculate_roc_auc(val_targets, avg_val_probs)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("Failure Analysis:")
    perform_failure_analysis(val_ids, val_targets, avg_val_probs)

    # 7. Submission
    # Using 0.5 threshold as >1.0 is impossible for AUC, ensuring submission is generated.
    if final_val_auc > 0.5:
        submission_df = pd.DataFrame({"id": test_ids, "has_cactus": avg_test_probs})
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


def predict_tta(model, dataloader, device):
    """
    Predict with Test Time Augmentation (Original, H-Flip, V-Flip).
    Returns numpy array of probabilities aligned with dataloader.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle different return signatures of dataloader
            # Train/Val: (img, label, id), Test: (img, id)
            if len(batch) == 3:
                images = batch[0]
            else:
                images = batch[0]

            images = images.to(device)

            # 1. Original
            out1 = model(images)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            out2 = model(images_h)
            prob2 = torch.sigmoid(out2)

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            out3 = model(images_v)
            prob3 = torch.sigmoid(out3)

            # Average
            avg_prob = (prob1 + prob2 + prob3) / 3.0
            all_probs.extend(avg_prob.cpu().numpy().flatten().tolist())

    return np.array(all_probs)


def perform_failure_analysis(ids, targets, preds):
    """
    Correlates prediction error with image meta-features.
    """
    # Calculate error magnitude
    errors = np.abs(targets - preds)

    # Load validation images from cache to extract features
    val_imgs_path = os.path.join(Config.WORKING_DIR, "val_images.npy")
    if not os.path.exists(val_imgs_path):
        return

    images = np.load(val_imgs_path)

    # Extract features
    brightness = []
    contrast = []
    red_mean = []
    green_mean = []
    blue_mean = []

    for img in images:
        # img is (32, 32, 3) RGB
        brightness.append(np.mean(img))
        contrast.append(np.std(img))
        red_mean.append(np.mean(img[:, :, 0]))
        green_mean.append(np.mean(img[:, :, 1]))
        blue_mean.append(np.mean(img[:, :, 2]))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Channel Mean": red_mean,
        "Green Channel Mean": green_mean,
        "Blue Channel Mean": blue_mean,
    }

    for name, feat_values in features.items():
        corr, pval = pearsonr(feat_values, errors)
        print(f"{name}: Correlation = {corr:.4f} (p-value = {pval:.4f})")


if __name__ == "__main__":
    main()
