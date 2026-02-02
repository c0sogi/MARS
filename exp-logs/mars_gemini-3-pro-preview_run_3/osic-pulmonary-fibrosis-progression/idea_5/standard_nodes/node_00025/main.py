import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, score_metric
from library.data import get_dataloaders, prepare_submission_df, LungDataset
from library.model import LateFusionNet, train_epoch, validate


def run_inference(model, loader, device):
    """
    Runs inference on a loader and returns vectors for analysis/submission.
    """
    model.eval()

    mus = []
    sigmas = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)

            # Predict
            mu_z, sigma_z = model(img, weeks, base_fvc, age, sex, smoke)

            # Inverse Transform
            mu_ml = mu_z * Config.FVC_STD + Config.FVC_MEAN
            sigma_ml = sigma_z * Config.FVC_STD

            # Collect results (move to CPU numpy)
            mus.extend(mu_ml.cpu().numpy().flatten())
            sigmas.extend(sigma_ml.cpu().numpy().flatten())

            if "target" in batch:
                target_z = batch["target"].to(device)
                target_ml = target_z * Config.FVC_STD + Config.FVC_MEAN
                targets.extend(target_ml.cpu().numpy().flatten())

    return np.array(mus), np.array(sigmas), np.array(targets)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print(f"Starting Runfile Execution. Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    # Fast baseline: Use full data but standard epochs (35 is fast for this data size)
    train_loader, val_loader = get_dataloaders(train_df, val_df)

    # 3. Model Setup
    model = LateFusionNet().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_score = validate(model, val_loader, Config.DEVICE)

        scheduler.step()

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Training Complete. Best Validation Score during training: {best_score}")

    # 5. Evaluation & Failure Analysis
    print("\n--- Evaluation & Failure Analysis ---")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    # Run inference on validation set to get detailed predictions
    val_pred_mu, val_pred_sigma, val_targets = run_inference(
        model, val_loader, Config.DEVICE
    )

    # Compute Final Metric
    # Note: score_metric handles the clipping of sigma internally for scoring
    final_metric = score_metric(val_targets, val_pred_mu, val_pred_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Features
    abs_error = np.abs(val_targets - val_pred_mu)

    # Create a DataFrame for correlation analysis
    # We use the original val_df to get feature values.
    # Since val_loader has shuffle=False, rows should align.
    analysis_df = val_df.copy()

    # Ensure lengths match (drop_last=False in val_loader)
    if len(analysis_df) == len(abs_error):
        analysis_df["AbsError"] = abs_error

        # Encode categorical for correlation
        analysis_df["Sex_Code"] = analysis_df["Sex"].map({"Male": 0, "Female": 1})
        analysis_df["Smoking_Code"] = analysis_df["SmokingStatus"].map(
            {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        )

        # Select features to correlate
        features = ["Weeks", "Age", "Percent", "Sex_Code", "Smoking_Code"]
        correlations = (
            analysis_df[features + ["AbsError"]].corr()["AbsError"].drop("AbsError")
        )

        print("\nCorrelation between Absolute Error and Features:")
        print(correlations.sort_values(ascending=False))
    else:
        print(
            "Warning: Validation dataframe length mismatch with predictions. Skipping detailed correlation."
        )

    # 6. Conditional Submission
    THRESHOLD = -6.6997912217

    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")

        test_df = pd.read_csv(Config.TEST_META_PATH)
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Prepare submission dataframe
        sub_df = prepare_submission_df(test_df, sample_sub)

        # Create submission dataset/loader
        sub_ds = LungDataset(sub_df, mode="submission")
        sub_loader = DataLoader(
            sub_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            drop_last=False,
        )

        # Inference
        sub_mu, sub_sigma, _ = run_inference(model, sub_loader, Config.DEVICE)

        # Post-process Sigma for Submission (Hard Clip)
        sub_sigma = np.maximum(sub_sigma, Config.MIN_CONFIDENCE)

        # Assign to dataframe
        sub_df["FVC"] = sub_mu
        sub_df["Confidence"] = sub_sigma

        # Format and Save
        submission = sub_df[["Patient_Week", "FVC", "Confidence"]]
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
