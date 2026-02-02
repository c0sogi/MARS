import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library
from library import config, utils, preprocess, data, model, engine


def run_failure_analysis(runner, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between absolute error and input features.
    """
    runner.model.eval()

    errors = []
    metas = {"Weeks": [], "Percent": [], "Age": []}

    # Inverse scaler for unscaling inputs for analysis
    scaler = utils.InverseScaler()

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(runner.device)
            tabular = batch["tabular"].to(runner.device)
            weeks = batch["weeks"].to(runner.device)
            target_fvc = batch["target_fvc"].to(runner.device)

            # Forward
            params = runner.model(image, tabular)
            mu_scaled, sigma_scaled = runner.calculate_trajectory(
                params, tabular, weeks
            )

            # Unscale for error calculation
            mu_raw = scaler.inverse_scale_fvc(mu_scaled)
            target_raw = scaler.inverse_scale_fvc(target_fvc)

            # Calculate Absolute Error
            abs_error = torch.abs(target_raw - mu_raw).cpu().numpy().flatten()
            errors.extend(abs_error)

            # Extract Metadata from Tabular/Weeks
            # Tabular: [Base_FVC, Base_Percent, Base_Age, Sex, ...]
            # We need to unscale these to get meaningful correlations or just use scaled (correlation is invariant to linear scaling)
            # Using scaled values is sufficient for Pearson correlation.

            # Weeks
            metas["Weeks"].extend(weeks.cpu().numpy().flatten())

            # Percent (Index 1 in tabular)
            metas["Percent"].extend(tabular[:, 1].cpu().numpy())

            # Age (Index 2 in tabular)
            metas["Age"].extend(tabular[:, 2].cpu().numpy())

    print("\nFailure Analysis (Correlation with Absolute Error):")
    for feature_name, values in metas.items():
        if len(values) > 1:
            corr, _ = pearsonr(values, errors)
            print(f"  {feature_name}: {corr:.4f}")


def generate_submission(runner, submission_path):
    """
    Generates the submission file for the test set.
    """
    print("Generating submission...")

    # Get loader
    sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
    test_loader = data.get_submission_loader(sample_sub_path)

    runner.model.eval()
    scaler = utils.InverseScaler()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(runner.device)
            tabular = batch["tabular"].to(runner.device)
            weeks = batch["weeks"].to(runner.device)
            patient_ids = batch["patient_id"]
            raw_weeks_batch = batch["raw_weeks"]

            # Forward
            params = runner.model(image, tabular)
            mu_scaled, sigma_scaled = runner.calculate_trajectory(
                params, tabular, weeks
            )

            # Unscale
            mu_raw = scaler.inverse_scale_fvc(mu_scaled).cpu().numpy().flatten()
            sigma_raw = scaler.inverse_scale_sigma(sigma_scaled).cpu().numpy().flatten()

            # Post-process
            # Clip confidence at 70ml as per metric requirement
            sigma_raw = np.maximum(sigma_raw, 70.0)

            # Collect results
            for i in range(len(patient_ids)):
                p_id = patient_ids[i]
                week = int(raw_weeks_batch[i].item())
                patient_week = f"{p_id}_{week}"

                results.append(
                    {
                        "Patient_Week": patient_week,
                        "FVC": mu_raw[i],
                        "Confidence": sigma_raw[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Save
    save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Preprocessing
    # Ensures images are cached and slopes are calculated
    preprocess.run_preprocessing(load_cached_data=True)

    # 3. Data Loading
    print("Loading data...")
    train_loader, val_loader = data.get_train_val_loaders(debug=config.DEBUG)

    # 4. Model Initialization
    print("Initializing model...")
    net = model.TAPNet().to(device)
    optimizer = engine.get_optimizer(net)
    scheduler = engine.get_scheduler(optimizer, config.EPOCHS)

    runner = engine.TAPNetEngine(net, device, optimizer, scheduler)

    # 5. Training Loop
    print(f"Starting training for {config.EPOCHS} epochs...")
    best_score = -float("inf")

    for epoch in range(config.EPOCHS):
        train_metrics = runner.train_one_epoch(train_loader)
        val_metrics = runner.evaluate(val_loader)

        # Save best model
        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            runner.save_checkpoint("best_model.pth")

    # 6. Final Evaluation
    print("Training complete. Loading best model for evaluation...")
    runner.load_checkpoint("best_model.pth")

    final_metrics = runner.evaluate(val_loader)
    print(f"Final Validation Metric: {final_metrics['score']}")

    # 7. Failure Analysis
    run_failure_analysis(runner, val_loader)

    # 8. Submission
    # Threshold check
    THRESHOLD = -6.57744688338769
    if final_metrics["score"] > THRESHOLD:
        generate_submission(runner, config.SUBMISSION_DIR)
    else:
        print(
            f"Validation score {final_metrics['score']} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
