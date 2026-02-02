import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_test_loader
from library.model import RCRFNet
from library.train import LaplaceNLLLoss, train_one_epoch, validate


def predict_dataset(model, loader, device, is_test=False):
    """
    Runs inference on a dataloader and returns inverse-transformed predictions.
    """
    model.eval()
    all_mu = []
    all_sigma = []
    all_targets = []

    with torch.no_grad():
        for imgs, clinical, t_rel, targets in loader:
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            t_rel = t_rel.to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(imgs, clinical, t_rel)

            all_mu.append(mu_scaled.cpu().numpy())
            all_sigma.append(sigma_scaled.cpu().numpy())
            if not is_test:
                all_targets.append(targets.numpy())

    # Concatenate results
    mu_scaled = np.concatenate(all_mu).flatten()
    sigma_scaled = np.concatenate(all_sigma).flatten()

    # Inverse Transformation
    # mu_final = mu_scaled * std + mean
    mu_final = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # sigma_final = sigma_scaled * std
    sigma_final = sigma_scaled * Config.TARGET_STD

    if not is_test:
        targets_scaled = np.concatenate(all_targets).flatten()
        targets_final = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN
        return mu_final, sigma_final, targets_final
    else:
        return mu_final, sigma_final


def main():
    # 1. Setup
    # Override epochs for convergence - Cite solution_lesson_node_00009
    # Extended to 30 to ensure sufficient gradient updates for projection heads.
    Config.EPOCHS = 30

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create directories if they don't exist (handled by Config.setup but good to ensure)
    Config.setup()

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = RCRFNet().to(device)

    # 4. Optimizer & Scheduler
    # Differential Learning Rates
    backbone_params = list(model.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceNLLLoss()

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = -float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_score = validate(model, val_loader, device)
        scheduler.step()

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # Required Output
    print(f"Final Validation Metric: {best_score}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Get predictions
    val_pred_mu, val_pred_sigma, val_true = predict_dataset(model, val_loader, device)

    # Calculate Error Magnitude
    error_magnitude = np.abs(val_true - val_pred_mu)

    # Get Metadata features for correlation
    # Note: val_loader.dataset is a LungDataset, which has a .df attribute
    val_df = val_loader.dataset.df.copy()

    # Ensure alignment: The loader (shuffle=False) preserves order of dataset.df
    # However, if drop_last was True (it's default False), we'd have issues.
    # LungDataset resets index, so we can assign directly.
    val_df["Error"] = error_magnitude
    val_df["Predicted_FVC"] = val_pred_mu
    val_df["Predicted_Sigma"] = val_pred_sigma

    # Calculate Correlations
    features_to_analyze = ["Age", "Baseline_FVC", "Weeks", "Percent"]
    # Add Relative_Weeks if available
    if "Relative_Weeks" in val_df.columns:
        features_to_analyze.append("Relative_Weeks")

    print("Correlation between Absolute Error and Features:")
    correlations = val_df[features_to_analyze + ["Error"]].corr()["Error"].drop("Error")
    print(correlations)

    # 7. Submission
    threshold = -6.573619738753321
    if best_score > threshold:
        print(
            f"\nValidation score {best_score} > {threshold}. Generating submission..."
        )

        # Get Test Loader
        test_loader, submission_df = get_test_loader(
            batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )

        # Inference
        test_mu, test_sigma = predict_dataset(model, test_loader, device, is_test=True)

        # Post-processing
        # Clip confidence at 70ml as per metric definition
        test_sigma_clipped = np.maximum(test_sigma, 70)

        # Assign to DataFrame
        submission_df["FVC"] = test_mu
        submission_df["Confidence"] = test_sigma_clipped

        # Format output
        final_submission = submission_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        final_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score {best_score} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
