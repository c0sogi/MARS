import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DSPRNet
from library.train import train_epoch, validate, generate_submission


def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between absolute error and input features.
    """
    model.eval()

    all_errors = []
    all_features = []

    # Feature names corresponding to the tabular input order in LungDataset
    # Tabular structure: [base_fvc_scaled, t_rel, age_scaled, sex_enc, smoke_enc]
    feature_names = [
        "BaseFVC_Scaled",
        "RelativeTime",
        "Age_Scaled",
        "Sex_Code",
        "Smoking_Code",
    ]

    with torch.no_grad():
        for batch in loader:
            img, tab, target = batch
            img = img.to(device)
            tab = tab.to(device)
            target = target.to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(img, tab)

            # Unscale predictions and target for error calculation in ml
            mu_raw = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
            target_raw = target * Config.TARGET_STD + Config.TARGET_MEAN

            # Calculate Absolute Error
            error = torch.abs(target_raw - mu_raw)

            all_errors.append(error.cpu().numpy())
            all_features.append(tab.cpu().numpy())

    # Concatenate all batches
    if len(all_errors) > 0:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features, axis=0)

        # Create DataFrame for correlation
        df_data = {name: all_features[:, i] for i, name in enumerate(feature_names)}
        df_data["AbsError"] = all_errors

        df = pd.DataFrame(df_data)

        # Calculate correlations
        correlations = df.corr()["AbsError"].drop("AbsError")

        print("\n--- Failure Analysis: Correlation with Absolute Error ---")
        print(correlations)
        print("-------------------------------------------------------")
    else:
        print("No validation data found for failure analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # Override Config for Fast Baseline
    # Reducing epochs to ensure execution within time limits while maintaining convergence
    Config.EPOCHS = 15
    Config.T_MAX = 15

    # 2. Data Loading
    # load_cached_data=True uses preprocessed .npy files
    train_loader, val_loader, sub_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    device = Config.DEVICE
    model = DSPRNet().to(device)

    # 4. Optimizer & Scheduler
    # Separate backbone and head parameters for differential learning rates
    backbone_params_ids = list(map(id, model.backbone.parameters()))

    backbone_params = [
        p
        for p in model.parameters()
        if id(p) in backbone_params_ids and p.requires_grad
    ]
    head_params = [
        p
        for p in model.parameters()
        if id(p) not in backbone_params_ids and p.requires_grad
    ]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # 5. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_metric = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission Logic
    # Threshold defined in task
    SUBMISSION_THRESHOLD = -6.573619738753321

    if final_metric > SUBMISSION_THRESHOLD:
        generate_submission(model, sub_loader, device)


if __name__ == "__main__":
    main()
