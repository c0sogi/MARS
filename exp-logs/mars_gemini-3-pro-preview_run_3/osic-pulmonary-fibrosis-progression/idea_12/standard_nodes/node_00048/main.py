import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Override config for fast baseline execution
import library.config

library.config.EPOCHS = 15  # Reduce epochs for speed
library.config.DEBUG = False

from library.config import (
    DEVICE,
    EPOCHS,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
    CHECKPOINT_DIR,
    TARGET_MEAN,
    TARGET_STD,
    METADATA_DIR,
    INPUT_DIR,
    SUBMISSION_DIR,
    TIME_SCALE,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, TabularProcessor, process_patient_images
from library.model import RSTCNet
from library.train import LaplaceLogLikelihoodLoss, train_one_epoch, validate_one_epoch


def perform_failure_analysis(model, val_loader):
    """
    Analyzes model performance on validation set and prints correlations.
    """
    model.eval()
    all_errors = []

    # We will collect features to correlate: Baseline FVC (norm), Age (norm), Time
    # Tabular structure: [BaseFVC, Age, Sex0, Sex1, Smoke0, Smoke1, Smoke2]
    all_base_fvc = []
    all_age = []
    all_time = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            time = batch["time"].to(DEVICE)
            targets = batch["target"].to(DEVICE)

            mu, sigma = model(images, tabular, time)

            # Inverse transform for error calculation in ml
            mu_orig = mu.cpu().numpy() * TARGET_STD + TARGET_MEAN
            target_orig = targets.cpu().numpy() * TARGET_STD + TARGET_MEAN

            # Absolute Error
            errors = np.abs(target_orig - mu_orig)
            all_errors.append(errors)

            # Collect features (move to cpu numpy)
            tab_np = tabular.cpu().numpy()
            time_np = time.cpu().numpy()

            all_base_fvc.append(tab_np[:, 0:1])  # Index 0 is BaseFVC
            all_age.append(tab_np[:, 1:2])  # Index 1 is Age
            all_time.append(time_np)

    all_errors = np.concatenate(all_errors).flatten()
    all_base_fvc = np.concatenate(all_base_fvc).flatten()
    all_age = np.concatenate(all_age).flatten()
    all_time = np.concatenate(all_time).flatten()

    print("\n=== Failure Analysis ===")
    print(f"Mean Absolute Error: {np.mean(all_errors):.4f} ml")

    # Correlations
    corr_base = np.corrcoef(all_errors, all_base_fvc)[0, 1]
    corr_age = np.corrcoef(all_errors, all_age)[0, 1]
    corr_time = np.corrcoef(all_errors, all_time)[0, 1]

    print("Correlation between Error Magnitude and Features:")
    print(f"  Baseline FVC:  {corr_base:.4f}")
    print(f"  Age:           {corr_age:.4f}")
    print(f"  Relative Time: {corr_time:.4f}")
    print("========================\n")


def generate_submission(model, processor):
    """
    Generates submission.csv for the test set.
    """
    print("Generating submission...")

    # 1. Load Data
    sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 2. Parse Patient and Weeks from sample_submission
    # Format: ID..._Week
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sample_sub["Target_Week"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )

    # 3. Merge with metadata to get Baseline info
    # test_meta contains the baseline measurement for each patient
    # We rename columns to avoid confusion
    test_meta_renamed = test_meta.rename(
        columns={"Weeks": "Baseline_Week", "FVC": "Baseline_FVC"}
    )

    # Merge
    df = sample_sub.merge(test_meta_renamed, on="Patient", how="left")

    # 4. Prepare Features
    # We need to construct the dataframe expected by TabularProcessor.transform
    # The processor expects: Patient, Weeks, FVC, Age, Sex, SmokingStatus
    # Here 'Weeks' refers to the target week for prediction
    # 'FVC' is the target, we can fill with dummy 0

    df["Weeks"] = df["Target_Week"]
    df["FVC"] = 0  # Dummy

    # Transform using the fitted processor
    # Note: We need to reconstruct the logic slightly because transform expects
    # to look up baselines from the passed df itself if we used the standard method.
    # However, TabularProcessor._get_baselines calculates baselines from the input df.
    # To reuse the processor correctly, we should pass a DF that looks like training data.
    # But simpler: we manually apply the transform logic since we have the columns.

    # Manual Transform to ensure correctness with test structure
    features = []
    times = []

    sex_map = {"Male": 0, "Female": 1}
    smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    for _, row in df.iterrows():
        # Continuous (Standardized using processor stats)
        age_norm = (row["Age"] - processor.age_mean) / processor.age_std
        base_fvc_norm = (
            row["Baseline_FVC"] - processor.base_fvc_mean
        ) / processor.base_fvc_std

        # Categorical
        sex_vec = [0, 0]
        if row["Sex"] in sex_map:
            sex_vec[sex_map[row["Sex"]]] = 1

        smoke_vec = [0, 0, 0]
        if row["SmokingStatus"] in smoke_map:
            smoke_vec[smoke_map[row["SmokingStatus"]]] = 1

        feat_vec = [base_fvc_norm, age_norm] + sex_vec + smoke_vec
        features.append(feat_vec)

        # Relative Time
        rel_week = row["Target_Week"] - row["Baseline_Week"]
        times.append(rel_week * TIME_SCALE)

    features = np.array(features, dtype=np.float32)
    times = np.array(times, dtype=np.float32).reshape(-1, 1)

    # 5. Predict
    model.eval()
    predictions_mu = []
    predictions_sigma = []

    # Group by patient to load image once
    unique_patients = df["Patient"].unique()

    # Create a mapping from original index to prediction
    results = np.zeros((len(df), 2))  # mu, sigma

    with torch.no_grad():
        for pid in unique_patients:
            # Indices for this patient
            p_indices = df.index[df["Patient"] == pid].tolist()

            # Load Image
            # Use relative path from test_meta
            img_path = test_meta[test_meta["Patient"] == pid]["image_path"].values[0]
            img_data = process_patient_images(pid, img_path)  # (H, W, 3)
            img_tensor = (
                torch.tensor(img_data, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            # Process in batches if many weeks (though usually ~100)
            # We can just push all weeks for one patient at once
            p_feats = torch.tensor(features[p_indices], dtype=torch.float32).to(DEVICE)
            p_times = torch.tensor(times[p_indices], dtype=torch.float32).to(DEVICE)

            # Repeat image for batch
            batch_size = len(p_indices)
            p_imgs = img_tensor.repeat(batch_size, 1, 1, 1)

            mu, sigma = model(p_imgs, p_feats, p_times)

            results[p_indices, 0] = mu.cpu().numpy().flatten()
            results[p_indices, 1] = sigma.cpu().numpy().flatten()

    # 6. Inverse Transform
    pred_fvc = results[:, 0] * TARGET_STD + TARGET_MEAN
    pred_sigma = results[:, 1] * TARGET_STD

    # 7. Create Submission DataFrame
    sub_df = pd.DataFrame(
        {"Patient_Week": df["Patient_Week"], "FVC": pred_fvc, "Confidence": pred_sigma}
    )

    # Apply clipping for submission
    sub_df["Confidence"] = sub_df["Confidence"].apply(lambda x: max(x, 70))

    # Save
    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def main():
    seed_everything()

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True, debug=False)

    # Fit a processor on train data for later use in submission
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    processor = TabularProcessor()
    processor.fit(train_df)

    # 2. Model Initialization
    model = RSTCNet().to(DEVICE)

    # 3. Optimizer Setup
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
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()

        # Save best
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Metric during training: {best_metric}")

    # 5. Final Validation & Analysis
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(best_model_path))

    # Recalculate metric on full validation set to be sure and print required format
    _, final_metric = validate_one_epoch(model, val_loader, criterion, DEVICE)
    print(f"Final Validation Metric: {final_metric}")

    perform_failure_analysis(model, val_loader)

    # 6. Submission Logic
    THRESHOLD = -6.57744688338769
    if final_metric > THRESHOLD:
        generate_submission(model, processor)
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
