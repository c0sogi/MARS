import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.utils import seed_everything, calculate_metric
from library.dataset import LungDataset
from library.architecture import AttentionFusedDualAxisNet
from library.engine import fit
from library.loss import ModifiedLaplaceLoss


def main():
    # 1. Configuration and Setup
    print("Initializing configuration...")
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 10  # Fast baseline: limited epochs
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    seed_everything(SEED)

    # 2. Data Loading
    print("Loading datasets...")
    # Initialize datasets
    # Note: load_cached=True allows utilizing pre-processed .npy files if they exist
    train_dataset = LungDataset(
        mode="train", cache_dir=os.path.join(WORKING_DIR, "idea_5"), load_cached=True
    )
    val_dataset = LungDataset(
        mode="val", cache_dir=os.path.join(WORKING_DIR, "idea_5"), load_cached=True
    )
    test_dataset = LungDataset(
        mode="test", cache_dir=os.path.join(WORKING_DIR, "idea_5"), load_cached=True
    )

    # Create DataLoaders
    # num_workers=2 to speed up data loading, pin_memory=True for GPU transfer speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 3. Model Initialization
    print("Initializing model...")
    model = AttentionFusedDualAxisNet(
        tabular_input_dim=6, feature_dim=1280, num_heads=4, pretrained=True
    )
    model.to(DEVICE)

    # 4. Training
    print("Starting training...")
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Using the provided engine.fit function
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=EPOCHS,
        patience=5,
        save_path=BEST_MODEL_PATH,
    )

    # 5. Validation and Failure Analysis
    print("Running validation inference...")
    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    val_results = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            time = batch["time"].to(DEVICE)
            target_fvc = batch["fvc"].to(DEVICE)
            baseline_fvc = batch["baseline_fvc"].to(DEVICE)
            patient_weeks = batch["patient_week"]

            # Forward pass
            alpha, sigma_base, sigma_growth = model(images, tabular)

            # Predict
            pred_fvc = baseline_fvc + alpha * time
            pred_sigma = sigma_base + sigma_growth * torch.abs(time)

            # Move to CPU for storage
            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()
            target_fvc_np = target_fvc.cpu().numpy()
            time_np = time.cpu().numpy()

            for i in range(len(patient_weeks)):
                val_results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "True_FVC": target_fvc_np[i],
                        "Pred_FVC": pred_fvc_np[i],
                        "Pred_Sigma": pred_sigma_np[i],
                        "Time_Delta": time_np[i],
                    }
                )

    val_df = pd.DataFrame(val_results)

    # Calculate Metric
    # Metric calculation requires arrays
    final_metric = calculate_metric(
        val_df["True_FVC"].values,
        val_df["Pred_FVC"].values,
        val_df["Pred_Sigma"].values,
    )

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    val_df["Error"] = np.abs(val_df["True_FVC"] - val_df["Pred_FVC"])

    # Merge with metadata to get raw features for correlation
    val_meta = pd.read_csv("./metadata/val.csv")
    # Create a key to merge
    val_meta["Patient_Week"] = val_meta["Patient"] + "_" + val_meta["Weeks"].astype(str)

    # Merge analysis df with metadata
    analysis_df = pd.merge(
        val_df,
        val_meta[["Patient_Week", "Age", "Percent", "Weeks"]],
        on="Patient_Week",
        how="left",
    )

    # Calculate correlations
    correlations = (
        analysis_df[["Error", "Age", "Percent", "Weeks", "Time_Delta"]]
        .corr()["Error"]
        .sort_values(ascending=False)
    )
    print("Correlation between Absolute Error and Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_results = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(DEVICE)
                tabular = batch["tabular"].to(DEVICE)
                time = batch["time"].to(DEVICE)
                baseline_fvc = batch["baseline_fvc"].to(DEVICE)
                patient_weeks = batch["patient_week"]

                # Forward pass
                alpha, sigma_base, sigma_growth = model(images, tabular)

                # Predict
                pred_fvc = baseline_fvc + alpha * time
                pred_sigma = sigma_base + sigma_growth * torch.abs(time)

                pred_fvc_np = pred_fvc.cpu().numpy()
                pred_sigma_np = pred_sigma.cpu().numpy()

                for i in range(len(patient_weeks)):
                    test_results.append(
                        {
                            "Patient_Week": patient_weeks[i],
                            "FVC": pred_fvc_np[i],
                            "Confidence": pred_sigma_np[i],
                        }
                    )

        submission_df = pd.DataFrame(test_results)

        # Ensure format
        submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(submission_df.head())

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
