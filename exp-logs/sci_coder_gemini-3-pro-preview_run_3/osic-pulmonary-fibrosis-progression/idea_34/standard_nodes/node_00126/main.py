import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library import utils, data, model, engine


def analyze_failures(net, val_loader, device, scaler_stats):
    """
    Performs failure analysis on the validation set by correlating
    prediction error with input features.
    """
    print("\n--- Failure Analysis ---")
    net.eval()

    errors = []
    feature_lists = {"Base_FVC": [], "Rel_Time": [], "Age": [], "Sex": [], "Smoke": []}

    fvc_mean = scaler_stats["fvc_mean"]
    fvc_std = scaler_stats["fvc_std"]

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            pred_mu, _ = net(tabular, imgs)

            # Unnormalize to ml
            pred_mu_ml = pred_mu * fvc_std + fvc_mean
            target_ml = targets * fvc_std + fvc_mean

            # Calculate Absolute Error
            batch_errors = torch.abs(target_ml - pred_mu_ml).cpu().numpy()
            errors.extend(batch_errors)

            # Extract features (Tabular is [Base_FVC, Rel_Time, Age, Sex, Smoke])
            # Move to CPU numpy
            tab_np = tabular.cpu().numpy()

            feature_lists["Base_FVC"].extend(tab_np[:, 0])
            feature_lists["Rel_Time"].extend(tab_np[:, 1])
            feature_lists["Age"].extend(tab_np[:, 2])
            feature_lists["Sex"].extend(tab_np[:, 3])
            feature_lists["Smoke"].extend(tab_np[:, 4])

    errors = np.array(errors)

    print(f"Mean Absolute Error on Validation: {np.mean(errors):.4f} ml")
    print("Correlation between Absolute Error and Features:")

    for feat_name, feat_vals in feature_lists.items():
        feat_vals = np.array(feat_vals)
        if len(np.unique(feat_vals)) > 1:
            corr, _ = pearsonr(feat_vals, errors)
            print(f"  {feat_name}: {corr:.4f}")
        else:
            print(f"  {feat_name}: N/A (Constant value)")


def generate_submission(net, device, scaler_stats):
    """
    Generates the submission file for the test set.
    """
    print("\n--- Generating Submission ---")

    # Get loader
    sub_loader, sub_df = data.get_submission_loader(scaler_stats, load_cached_data=True)

    net.eval()
    predictions_fvc = []
    predictions_conf = []

    fvc_mean = scaler_stats["fvc_mean"]
    fvc_std = scaler_stats["fvc_std"]

    with torch.no_grad():
        for batch in sub_loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)

            # Forward
            pred_mu, pred_sigma = net(tabular, imgs)

            # Unnormalize
            # mu_final = mu_scaled * sigma_target + mu_target
            pred_mu_ml = pred_mu * fvc_std + fvc_mean

            # sigma_final = sigma_scaled * sigma_target
            pred_sigma_ml = pred_sigma * fvc_std

            predictions_fvc.extend(pred_mu_ml.cpu().numpy())
            predictions_conf.extend(pred_sigma_ml.cpu().numpy())

    # Post-processing
    # 1. Clip Confidence at 70ml
    predictions_conf = np.maximum(predictions_conf, 70)

    # 2. Assign to DataFrame
    # sub_loader does not shuffle, so order matches sub_df
    sub_df["FVC"] = predictions_fvc
    sub_df["Confidence"] = predictions_conf

    # 3. Format Submission
    submission = sub_df[["Patient_Week", "FVC", "Confidence"]].copy()

    # Ensure types
    submission["FVC"] = submission["FVC"].astype(float)
    submission["Confidence"] = submission["Confidence"].astype(float)

    # Save
    Config.mkdirs()  # Ensure dirs exist
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())


def main():
    # 1. Setup
    utils.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Modify Config for Fast Baseline
    # Cite solution_lesson_node_00100: Scheduler horizon coupled to training duration
    Config.EPOCHS = (
        25  # Increased slightly to ensure convergence (Lesson 52: converged ~14)
    )

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, scaler_stats = data.get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    net = model.DSPRNet().to(device)

    # Differential Learning Rates
    optimizer = torch.optim.AdamW(
        [
            # Stream A (Linear) and Heads get higher LR
            {"params": net.linear_stream.parameters(), "lr": Config.LR_HEADS},
            {"params": net.deep_head.parameters(), "lr": Config.LR_HEADS},
            {"params": net.final_head.parameters(), "lr": Config.LR_HEADS},
            # Stream B (Backbone) gets lower LR
            {
                "params": net.backbone.parameters(),
                "lr": Config.LR_BACKBONE,
            },
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training
    print("Starting Training...")
    best_score = engine.run_training(
        net,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        scaler_stats,
        epochs=Config.EPOCHS,
        patience=10,
    )

    # 5. Final Evaluation
    print("Loading Best Model for Evaluation...")
    net.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    final_metric = engine.evaluate(net, val_loader, device, scaler_stats)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(net, val_loader, device, scaler_stats)

    # 7. Submission Logic
    threshold = -6.573619738753321
    if final_metric > threshold:
        generate_submission(net, device, scaler_stats)
    else:
        print(f"\nSkipping submission. Metric {final_metric} <= Threshold {threshold}")


if __name__ == "__main__":
    main()
