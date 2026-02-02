import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config, seed_everything
from library.data import (
    CTScanProcessor,
    TabularScaler,
    PulmonaryDataset,
    get_baseline_lookup,
    prepare_inference_dataframe,
)
from library.model import CAPNet
from library.train import train_one_epoch, evaluate
from library.utils import metric_laplace_log_likelihood


def main():
    # 1. Setup
    Config.setup()
    # Override epochs for fine-tuning strategy
    Config.EPOCHS = 50
    device = Config.DEVICE

    # Silent execution for libraries
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print(f"Initializing CAP-Net Pipeline on {device}...")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # 3. Initialize Processors
    processor = CTScanProcessor()
    scaler = TabularScaler()
    scaler.fit(train_df)

    # Baseline lookups
    train_lookup = get_baseline_lookup(train_df)
    val_lookup = get_baseline_lookup(val_df)

    # 4. Create Datasets and Loaders
    train_dataset = PulmonaryDataset(
        train_df, processor, scaler, train_lookup, mode="train"
    )
    val_dataset = PulmonaryDataset(val_df, processor, scaler, val_lookup, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Initialize Model
    model = CAPNet().to(device)

    # Differential Learning Rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.BACKBONE_LR},
            {"params": head_params, "lr": Config.HEAD_LR},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 6. Training Loop
    best_score = -float("inf")

    print(f"Training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = evaluate(model, val_loader, scaler, device)
        scheduler.step()

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training complete. Best Validation Score: {best_score}")

    # 7. Final Validation & Failure Analysis
    print("\nRunning Final Validation & Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    val_predictions = []
    val_sigmas = []
    val_targets = []
    val_meta_features = []

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            meta_cat = batch["meta_cat"].to(device)
            meta_num = batch["meta_num"].to(device)
            baseline_fvc_scaled = batch["baseline_fvc_scaled"].to(device)
            weeks_scaled = batch["weeks_scaled"].to(device)

            mu_scaled, sigma_scaled = model(
                image, meta_cat, meta_num, baseline_fvc_scaled, weeks_scaled
            )

            # Unscale
            mu_unscaled = scaler.unscale_fvc(mu_scaled.cpu().numpy())
            sigma_unscaled = scaler.unscale_sigma(sigma_scaled.cpu().numpy())
            raw_fvc = batch["raw_fvc"].numpy()

            val_predictions.extend(mu_unscaled)
            val_sigmas.extend(sigma_unscaled)
            val_targets.extend(raw_fvc)

            # Collect features for analysis
            # meta_num: [Age_scaled, Percent_scaled]
            # We need raw values for correlation, but scaled is fine for correlation magnitude checks
            # Let's grab raw from batch if possible, or just use scaled

            # Reconstruct features from batch data for analysis
            # We need to match the batch order
            # We can extract from the dataset using indices, but batch shuffling is off for val_loader
            pass

    # Calculate Metric
    val_predictions = np.array(val_predictions)
    val_sigmas = np.array(val_sigmas)
    val_targets = np.array(val_targets)

    final_metric = metric_laplace_log_likelihood(
        val_targets, val_predictions, val_sigmas
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    errors = np.abs(val_targets - val_predictions)

    # Construct a dataframe for analysis matching the validation set order
    analysis_df = val_df.copy()
    analysis_df["Error"] = errors

    # Add Baseline FVC to analysis df
    analysis_df["Baseline_FVC"] = analysis_df["Patient"].map(val_lookup)

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    features_to_analyze = ["Weeks", "Age", "Percent", "Baseline_FVC"]
    for feat in features_to_analyze:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["Error"])
            print(f"  Error vs {feat}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = -6.57744688338769
    if final_metric > THRESHOLD:
        print("\nGenerating Submission...")

        # Prepare Test Data
        test_df = prepare_inference_dataframe(Config.SAMPLE_SUBMISSION, Config.TEST_CSV)

        # Test Baseline Lookup (from test metadata)
        # prepare_inference_dataframe puts baseline FVC in 'Base_FVC'
        # We need a lookup for the dataset class
        test_lookup = dict(zip(test_df["Patient"], test_df["Base_FVC"]))

        # Create Test Dataset
        test_dataset = PulmonaryDataset(
            test_df, processor, scaler, test_lookup, mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_preds = []
        all_sigmas = []
        patient_weeks = []

        with torch.no_grad():
            for batch in test_loader:
                image = batch["image"].to(device)
                meta_cat = batch["meta_cat"].to(device)
                meta_num = batch["meta_num"].to(device)
                baseline_fvc_scaled = batch["baseline_fvc_scaled"].to(device)
                weeks_scaled = batch["weeks_scaled"].to(device)

                mu_scaled, sigma_scaled = model(
                    image, meta_cat, meta_num, baseline_fvc_scaled, weeks_scaled
                )

                mu_unscaled = scaler.unscale_fvc(mu_scaled.cpu().numpy())
                sigma_unscaled = scaler.unscale_sigma(sigma_scaled.cpu().numpy())

                all_preds.extend(mu_unscaled)
                all_sigmas.extend(sigma_unscaled)
                patient_weeks.extend(batch["patient_week"])

        # Format Submission
        # Clip confidence as per metric requirement
        all_sigmas = np.maximum(all_sigmas, 70)

        submission = pd.DataFrame(
            {"Patient_Week": patient_weeks, "FVC": all_preds, "Confidence": all_sigmas}
        )

        # Ensure FVC is valid (positive)
        submission["FVC"] = np.maximum(submission["FVC"], 0)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
