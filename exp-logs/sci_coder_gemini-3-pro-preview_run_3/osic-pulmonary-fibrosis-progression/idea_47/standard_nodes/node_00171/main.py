import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, metric_score, InverseScaler
from library.data import get_dataloaders
from library.model import DSPRNet
from library.loss import LaplaceNLLLoss
from library.train import train_one_epoch, validate


def run_inference(model, loader, device):
    """
    Runs inference on a loader and returns collected predictions and inputs.
    """
    model.eval()
    preds_mean_list = []
    preds_sigma_list = []
    targets_list = []
    tabular_list = []

    inverse_scaler = InverseScaler()

    with torch.no_grad():
        for batch in loader:
            # Handle cases where loader returns (img, tab, target) or (img, tab)
            if len(batch) == 3:
                imgs, tabular, targets = batch
                targets = targets.to(device)
                targets_list.append(targets.cpu().numpy())
            else:
                imgs, tabular = batch

            imgs = imgs.to(device)
            tabular = tabular.to(device)
            tabular_list.append(tabular.cpu().numpy())

            # Forward pass
            out = model(imgs, tabular)

            # Extract and process outputs
            pred_mean_norm = out[:, 0]
            pred_raw_sigma = out[:, 1]
            pred_sigma_norm = F.softplus(pred_raw_sigma) + 1e-6

            # Inverse scale
            pred_mean_orig, pred_sigma_orig = inverse_scaler(
                pred_mean_norm, pred_sigma_norm
            )

            preds_mean_list.append(pred_mean_orig.cpu().numpy())
            preds_sigma_list.append(pred_sigma_orig.cpu().numpy())

    # Concatenate results
    preds_mean = np.concatenate(preds_mean_list)
    preds_sigma = np.concatenate(preds_sigma_list)
    tabular_data = np.concatenate(tabular_list)

    if targets_list:
        # Inverse scale targets
        targets_norm = np.concatenate(targets_list)
        mean_val, std_val = Config.get_target_stats()
        targets_orig = targets_norm * std_val + mean_val
        return preds_mean, preds_sigma, tabular_data, targets_orig
    else:
        return preds_mean, preds_sigma, tabular_data


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Initializing training on {device}...")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # 3. Model Initialization
    model = DSPRNet().to(device)

    # 4. Optimizer & Scheduler Setup
    # Group 1: Backbone parameters
    backbone_params = list(model.visual_encoder.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))
    # Group 2: Head parameters
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    criterion = LaplaceNLLLoss()

    # 5. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved.")

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions on validation set
    val_preds_mean, val_preds_sigma, val_tabular, val_targets = run_inference(
        model, val_loader, device
    )

    # Calculate Final Metric
    # Flatten arrays for metric calculation
    final_metric = metric_score(
        val_targets.flatten(), val_preds_mean.flatten(), val_preds_sigma.flatten()
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    abs_error = np.abs(val_targets.flatten() - val_preds_mean.flatten())

    # Tabular features: [Baseline_FVC_Scaled, Relative_Week_Scaled, Age_Scaled, Sex_Code, Smoking_Code]
    # We correlate error with these features
    features = {
        "Baseline_FVC": val_tabular[:, 0],
        "Relative_Week": val_tabular[:, 1],
        "Age": val_tabular[:, 2],
        "Sex": val_tabular[:, 3],
        "Smoking": val_tabular[:, 4],
        "True_FVC": val_targets.flatten(),
    }

    print("\nFailure Analysis (Correlation with Absolute Error):")
    for name, values in features.items():
        # Ensure shapes match
        if values.shape == abs_error.shape:
            corr = np.corrcoef(abs_error, values)[0, 1]
            print(f"  Error vs {name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print("\nMetric threshold passed. Generating submission...")

        # Inference on Test Set
        test_preds_mean, test_preds_sigma, _ = run_inference(model, test_loader, device)

        # Clip confidence at 70ml as per metric requirement
        test_preds_sigma = np.maximum(test_preds_sigma, 70)

        # Load sample submission to get IDs
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Ensure lengths match
        if len(sample_sub) != len(test_preds_mean):
            print(
                f"Warning: Prediction count {len(test_preds_mean)} does not match submission rows {len(sample_sub)}."
            )
            # In case of mismatch (e.g. drop_last in loader), we might need robust handling,
            # but test loader should be exact.

        # Assign values
        sample_sub["FVC"] = test_preds_mean
        sample_sub["Confidence"] = test_preds_sigma

        # Save
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sample_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric {final_metric} did not beat threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
