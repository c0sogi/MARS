import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, score_function, seed_everything
from library.data import get_dataloaders
from library.model import UDSRNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the negative of the modified Laplace Log Likelihood metric as a loss function.
    Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, pred_mu, pred_sigma, target):
        # Flatten target if necessary to match pred_mu shape (B,)
        if target.ndim > 1:
            target = target.view(-1)

        # Calculate Delta
        delta = torch.abs(target - pred_mu)

        # Constants
        sqrt_2 = np.sqrt(2)

        # Loss calculation
        # pred_sigma is guaranteed positive via softplus in the model
        loss = (sqrt_2 * delta) / pred_sigma + torch.log(sqrt_2 * pred_sigma)

        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        imgs = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["fvc_target"].to(device)

        # Forward pass
        optimizer.zero_grad()
        mu, sigma = model(imgs, tabular)

        # Loss calculation
        loss = criterion(mu, sigma, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), imgs.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device, target_scaler):
    model.eval()
    loss_meter = AverageMeter()
    score_meter = AverageMeter()

    # Scaler parameters for inverse transform
    scale = target_scaler.scale_[0]
    mean = target_scaler.mean_[0]

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            target_scaled = batch["fvc_target"].to(device)
            target_raw = batch["fvc_raw"].to(device)

            mu_scaled, sigma_scaled = model(imgs, tabular)

            # 1. Validation Loss (in scaled space)
            loss = criterion(mu_scaled, sigma_scaled, target_scaled)
            loss_meter.update(loss.item(), imgs.size(0))

            # 2. Metric Score (in original space)
            # Inverse transform predictions
            mu_real = mu_scaled.cpu().numpy() * scale + mean
            sigma_real = sigma_scaled.cpu().numpy() * scale
            y_true = target_raw.cpu().numpy()

            # Calculate metric
            batch_score = score_function(y_true, mu_real, sigma_real)
            score_meter.update(batch_score, imgs.size(0))

    return loss_meter.avg, score_meter.avg


def generate_submission(model, scalers, device):
    print("Generating submission...")
    model.eval()

    # Load required files
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Extract Patient and Week from Patient_Week
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = (
        sub_df["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
    )

    # Prepare Baseline Data
    # Rename columns to avoid collision and clarify meaning
    base_df = test_df.rename(
        columns={
            "Weeks": "Baseline_Week",
            "FVC": "Baseline_FVC",
            "Age": "Base_Age",
            "Sex": "Base_Sex",
            "SmokingStatus": "Base_Smoking",
        }
    )

    # Merge baseline info into submission dataframe
    df = sub_df.merge(base_df, on="Patient", how="left")

    # --- Feature Engineering (Replicating OSICDataset logic) ---

    # 1. Relative Time
    df["Relative_Time"] = (df["Weeks"] - df["Baseline_Week"]) * Config.TIME_SCALE

    # 2. Smoking Status Encoding
    smoking_enc = scalers["smoking_encoder"]
    smoking_reshaped = df["Base_Smoking"].values.reshape(-1, 1)
    df["Smoking_Code"] = smoking_enc.transform(smoking_reshaped).flatten()

    # 3. Sex Encoding
    df["Sex_Code"] = df["Base_Sex"].apply(lambda x: 0 if x == "Male" else 1)

    # 4. Input Scaling (Age, Baseline FVC)
    std_scaler = scalers["standard_scaler"]
    input_feats = df[["Base_Age", "Baseline_FVC"]].values
    scaled_feats = std_scaler.transform(input_feats)
    df["Age_Scaled"] = scaled_feats[:, 0]
    df["Baseline_FVC_Scaled"] = scaled_feats[:, 1]

    # --- Inference ---

    # Target Scaler for inverse transform
    target_scaler = scalers["target_scaler"]
    scale = target_scaler.scale_[0]
    mean = target_scaler.mean_[0]

    result_dfs = []
    unique_patients = df["Patient"].unique()

    for patient in unique_patients:
        # Get patient subset
        pat_df = df[df["Patient"] == patient].copy()

        # Load Image (Cached)
        img_path = os.path.join(Config.CACHE_DIR, f"{patient}.npy")
        if os.path.exists(img_path):
            img_vol = np.load(img_path)
        else:
            # Fallback (should not happen if cache is built)
            img_vol = np.zeros(
                (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32,
            )

        # Prepare Image Tensor: (1, C, H, W) -> (B, C, H, W)
        img_tensor = torch.tensor(img_vol, dtype=torch.float32).unsqueeze(0).to(device)

        # Prepare Tabular Tensor: (B, 5)
        # Cols: [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]
        tab_data = pat_df[
            [
                "Baseline_FVC_Scaled",
                "Relative_Time",
                "Age_Scaled",
                "Sex_Code",
                "Smoking_Code",
            ]
        ].values
        tab_tensor = torch.tensor(tab_data, dtype=torch.float32).to(device)

        # Expand image to batch size
        batch_size = tab_tensor.size(0)
        img_batch = img_tensor.expand(batch_size, -1, -1, -1)

        # Predict
        with torch.no_grad():
            mu_scaled, sigma_scaled = model(img_batch, tab_tensor)

        # Inverse Transform
        mu_real = mu_scaled.cpu().numpy() * scale + mean
        sigma_real = sigma_scaled.cpu().numpy() * scale

        # Store results
        pat_df["FVC_Pred"] = mu_real
        pat_df["Confidence_Pred"] = sigma_real
        result_dfs.append(pat_df)

    # Concatenate results
    final_df = pd.concat(result_dfs)

    # --- Formatting ---

    # Clip Confidence
    final_df["Confidence"] = np.maximum(
        final_df["Confidence_Pred"], Config.CONFIDENCE_CLIP
    )
    final_df["FVC"] = final_df["FVC_Pred"]

    # Select columns
    submission = final_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader, scalers = get_dataloaders()
    target_scaler = scalers["target_scaler"]

    # Model
    print("Initializing model...")
    model = UDSRNet().to(device)

    # Optimizer (Differential Learning Rates)
    # Filter parameters that require gradients
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss
    criterion = LaplaceLogLikelihoodLoss()

    # Training Loop
    best_score = -float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate(
            model, val_loader, criterion, device, target_scaler
        )

        scheduler.step()

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved! Score: {best_score}")

    print(f"Training complete. Best Validation Score: {best_score}")

    # Generate Submission
    # Load best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    generate_submission(model, scalers, device)
