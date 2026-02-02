import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, load_checkpoint, compute_metric
from library.dataset import LungDataset
from library.model import NSLHN
from library.engine import fit


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger()

    # Adjust Config for Fast Baseline execution
    # We limit epochs to 20 to ensure completion within 2 hours
    Config.EPOCHS = 20
    Config.print_config()

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    logger.info("Initializing Datasets...")
    train_dataset = LungDataset(mode="train", split="train")
    val_dataset = LungDataset(mode="inference", split="val")

    # Use num_workers from Config, pin_memory for GPU speed
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

    # 3. Model Initialization
    logger.info("Initializing NSL-HN Model...")
    model = NSLHN().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training
    logger.info("Starting Training Loop...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    # 5. Validation & Failure Analysis
    logger.info("Loading best model for analysis...")
    best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    load_checkpoint(best_ckpt_path, model)
    model.eval()

    # Run Inference on Validation Set
    val_true_fvc = []
    val_pred_fvc = []
    val_pred_sigma = []

    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            fvc_target = batch["fvc_target"].to(device)

            p_fvc, p_sigma = model(axial, coronal, tabular, base_fvc, delta_week)

            val_true_fvc.extend(fvc_target.cpu().numpy())
            val_pred_fvc.extend(p_fvc.cpu().numpy())
            val_pred_sigma.extend(p_sigma.cpu().numpy())

    val_true_fvc = np.array(val_true_fvc)
    val_pred_fvc = np.array(val_pred_fvc)
    val_pred_sigma = np.array(val_pred_sigma)

    # Compute Metric
    final_metric = compute_metric(val_true_fvc, val_pred_fvc, val_pred_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    abs_errors = np.abs(val_true_fvc - val_pred_fvc)

    # We use the underlying dataframe from the validation dataset
    # Since shuffle=False, the order matches our predictions
    analysis_df = val_dataset.df.copy()
    analysis_df["Abs_Error"] = abs_errors
    analysis_df["Pred_FVC"] = val_pred_fvc
    analysis_df["Pred_Sigma"] = val_pred_sigma

    # Select features for correlation
    features_to_corr = ["Age", "Percent", "Weeks", "Base_FVC", "Base_Percent"]
    # Ensure these columns exist (dataset preparation might rename them or keep them)
    # LungDataset creates Base_ columns, but original might still be there.
    # Let's check what's available based on dataset.py:
    # It has 'Weeks', 'Percent', 'Age', 'Base_FVC', 'Base_Percent', 'Base_Age'

    avail_features = [f for f in features_to_corr if f in analysis_df.columns]

    if avail_features:
        correlations = analysis_df[avail_features].corrwith(analysis_df["Abs_Error"])
        print("\nCorrelation between Absolute Error and Features:")
        print(correlations)
    else:
        logger.warning("Features for correlation analysis not found in DataFrame.")

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = LungDataset(mode="inference", split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_ids = []
        submission_fvc = []
        submission_conf = []

        with torch.no_grad():
            for batch in test_loader:
                axial = batch["axial"].to(device)
                coronal = batch["coronal"].to(device)
                tabular = batch["tabular"].to(device)
                base_fvc = batch["base_fvc"].to(device)
                delta_week = batch["delta_week"].to(device)
                pw_ids = batch["patient_week_id"]  # List of strings

                p_fvc, p_sigma = model(axial, coronal, tabular, base_fvc, delta_week)

                # Clip confidence as per metric requirement (though metric func does it,
                # submission usually expects raw, but let's be safe and output valid values)
                # The metric definition says "confidence values are clipped at 70 ml".
                # The model outputs raw sigma. We will output raw sigma, but ensure it's positive.
                # The model uses softplus, so it is positive.

                submission_ids.extend(pw_ids)
                submission_fvc.extend(p_fvc.cpu().numpy())
                submission_conf.extend(p_sigma.cpu().numpy())

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "Patient_Week": submission_ids,
                "FVC": submission_fvc,
                "Confidence": submission_conf,
            }
        )

        # Save
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        logger.warning(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
