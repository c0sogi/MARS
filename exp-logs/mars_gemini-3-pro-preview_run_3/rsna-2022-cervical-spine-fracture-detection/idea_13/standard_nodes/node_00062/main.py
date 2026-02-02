import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library import utils, dataset, model, engine


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    # Set random seeds for reproducibility
    utils.seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    Config.EPOCHS = (
        10  # Increased to 10 for better convergence. Cite solution_lesson_node_00007
    )
    Config.NUM_WORKERS = 8  # Maximize I/O throughput (12 vCPUs available)

    # Ensure device is set correctly
    device = Config.DEVICE
    print(f"Running on device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing Datasets...")

    # Training Dataset
    train_ds = dataset.RSNADataset(subset="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain stability
    )

    # Validation Dataset
    val_ds = dataset.RSNADataset(subset="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # =========================================================================
    # 3. Model Initialization & Training
    # =========================================================================
    print("Initializing Dual-Stream ConvNeXt Model...")
    net = model.DualStreamConvNeXt()
    net.to(device)

    print("Starting Training Loop...")
    # engine.fit handles the training loop, validation, and saving the best model
    engine.fit(net, train_loader, val_loader, device)

    # =========================================================================
    # 4. Validation Assessment & Failure Analysis
    # =========================================================================
    print("\n=== Validation Assessment ===")

    # Load the best model weights
    best_model_path = os.path.join("working", "best_model.pth")
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model not found. Using current weights.")

    net.eval()

    # Collect predictions and targets for full analysis
    all_preds = []
    all_targets = []

    # We also want to extract metadata features for failure analysis
    val_df = val_ds.df
    num_slices_list = []

    # Pre-calculate num_slices for correlation analysis
    # We do this by checking file counts in the directories listed in metadata
    for _, row in val_df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["image_path"])
        if os.path.exists(path):
            # Count files (slices)
            n_slices = len(
                [f for f in os.listdir(path) if f.endswith(".dcm") or f.isdigit()]
            )
            num_slices_list.append(n_slices)
        else:
            num_slices_list.append(0)

    print("Generating validation predictions for analysis...")
    with torch.no_grad():
        for global_input, targets in val_loader:
            global_input = global_input.to(device)

            logits = net(global_input)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate and Print Final Metric
    final_metric = utils.competition_metric(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")

    # Calculate weighted log loss per sample
    # Weights: 1/7 for C1-C7, 1.0 for patient_overall
    weights = np.array([1 / 7] * 7 + [1.0])
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    sample_losses = []
    for i in range(len(all_targets)):
        t = all_targets[i]
        p = preds_clipped[i]
        # Binary Cross Entropy per class
        bce = -(t * np.log(p) + (1 - t) * np.log(1 - p))
        # Weighted average loss for this sample
        w_loss = np.sum(bce * weights) / np.sum(weights)
        sample_losses.append(w_loss)

    sample_losses = np.array(sample_losses)
    num_slices_arr = np.array(num_slices_list)

    # 1. Correlation with Input Feature: Number of Slices
    if len(sample_losses) > 1 and np.std(num_slices_arr) > 0:
        corr_slices = np.corrcoef(sample_losses, num_slices_arr)[0, 1]
        print(f"Correlation (Error Magnitude vs Num Slices): {corr_slices:.4f}")
    else:
        print(
            "Correlation (Error Magnitude vs Num Slices): N/A (Insufficient variance)"
        )

    # 2. Correlation with Target: Patient Overall Fracture
    patient_overall_target = all_targets[:, 7]
    if len(sample_losses) > 1 and np.std(patient_overall_target) > 0:
        corr_fracture = np.corrcoef(sample_losses, patient_overall_target)[0, 1]
        print(
            f"Correlation (Error Magnitude vs Fracture Presence): {corr_fracture:.4f}"
        )
    else:
        print("Correlation (Error Magnitude vs Fracture Presence): N/A")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    THRESHOLD = 0.06429807151236185

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Initialize Test Dataset
        test_ds = dataset.RSNADataset(subset="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Inference
        engine.inference(net, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
