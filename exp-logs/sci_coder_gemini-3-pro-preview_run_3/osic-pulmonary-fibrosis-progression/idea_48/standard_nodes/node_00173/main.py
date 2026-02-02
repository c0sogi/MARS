import sys
import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, inverse_transform, calculate_metric
from library.data import get_dataloaders
from library.model import CASDSNet
from library.train import LLLLoss, train_epoch, val_epoch


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Fast Baseline Overrides
    Config.EPOCHS = 30
    Config.BATCH_SIZE = 32

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # 3. Model Initialization
    model = CASDSNet().to(device)

    # Differential Learning Rates
    backbone_params = list(model.image_encoder.parameters())
    backbone_ids = list(map(id, backbone_params))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = LLLLoss()

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print("Training complete.")

    # 5. Final Validation & Failure Analysis
    print("Performing validation analysis...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []
    val_tabular_inputs = []

    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device).squeeze(-1)

            mu, sigma = model(images, tabular)

            # Inverse transform predictions
            mu_orig, sigma_orig = inverse_transform(mu, sigma)

            # Inverse transform targets
            target_orig = targets.cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN

            val_preds_mu.extend(mu_orig)
            val_preds_sigma.extend(sigma_orig)
            val_targets.extend(target_orig)
            val_tabular_inputs.extend(tabular.cpu().numpy())

    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)
    val_tabular_inputs = np.array(val_tabular_inputs)

    # Calculate and Print Final Metric
    final_metric = calculate_metric(val_targets, val_preds_mu, val_preds_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Tabular features: [Baseline_FVC_std, t_rel, Age_std, Sex, Smoke]
    errors = np.abs(val_targets - val_preds_mu)

    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Baseline_FVC_Std": val_tabular_inputs[:, 0],
            "Relative_Time": val_tabular_inputs[:, 1],
            "Age_Std": val_tabular_inputs[:, 2],
            "Sex": val_tabular_inputs[:, 3],
            "Smoking": val_tabular_inputs[:, 4],
        }
    )

    print("Failure Analysis (Correlation with Absolute Error):")
    correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
    print(correlations)

    # 6. Submission Generation
    threshold = -6.573619738753321
    if final_metric > threshold:
        print(f"Metric {final_metric} > {threshold}. Generating submission...")

        test_preds_mu = []
        test_preds_sigma = []

        with torch.no_grad():
            for images, tabular, _ in test_loader:
                images = images.to(device)
                tabular = tabular.to(device)

                mu, sigma = model(images, tabular)
                mu_orig, sigma_orig = inverse_transform(mu, sigma)

                test_preds_mu.extend(mu_orig)
                test_preds_sigma.extend(sigma_orig)

        # Create Submission DataFrame
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Safety check for length
        if len(sub_df) == len(test_preds_mu):
            sub_df["FVC"] = test_preds_mu
            sub_df["Confidence"] = test_preds_sigma

            # Post-processing: Clip Confidence at 70ml
            sub_df["Confidence"] = sub_df["Confidence"].apply(lambda x: max(x, 70))

            save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
            sub_df.to_csv(save_path, index=False)
            print(f"Submission saved to {save_path}")
        else:
            print(
                f"Error: Submission length mismatch. DF: {len(sub_df)}, Preds: {len(test_preds_mu)}"
            )
    else:
        print(f"Metric {final_metric} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
