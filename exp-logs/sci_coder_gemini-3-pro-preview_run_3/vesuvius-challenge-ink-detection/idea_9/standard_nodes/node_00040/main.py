import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Import library modules
from library.config import Config
from library import data_utils, model, loss, train_utils, inference_utils


def main():
    # 1. Setup
    # -------------------------------------------------------------------------
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("Initializing Datasets...")

    # Training Dataset
    # We use load_cached_data=True to utilize pre-processed .npy files if available
    train_dataset = data_utils.InkDataset(
        split="train", mode="train", load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Dataset (Patch-based for monitoring loss)
    val_dataset = data_utils.InkDataset(
        split="val",
        mode="val",
        load_cached_data=True,
        normalization_stats=(train_dataset.mean, train_dataset.std),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches per epoch: {len(train_loader)}")
    print(
        f"Global Normalization Stats - Mean: {train_dataset.mean:.4f}, Std: {train_dataset.std:.4f}"
    )

    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing DilatedFCN Model...")
    net = model.DilatedFCN().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = loss.BCEDiceLoss()

    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting Training...")
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_utils.train_one_epoch(
            net, train_loader, optimizer, criterion, device
        )

        # Validate (Patch-level loss)
        val_loss, _, _ = train_utils.evaluate(net, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), Config.MODEL_PATH)
            # print("  -> Saved best model")

    print("Training Complete.")

    # 5. Full Validation & Threshold Optimization
    # -------------------------------------------------------------------------
    print("Running Full Validation Inference...")

    # Load best model for inference
    net.load_state_dict(torch.load(Config.MODEL_PATH))
    net.eval()

    # Get validation fragment IDs
    val_meta_path = Config.METADATA_DIR / "val.csv"
    if val_meta_path.exists():
        val_meta = pd.read_csv(val_meta_path)
        val_ids = val_meta["fragment_id"].astype(str).tolist()
    else:
        print("Validation metadata not found. Skipping validation.")
        val_ids = []

    all_preds_flat = []
    all_labels_flat = []

    # Store data for failure analysis
    analysis_data = []

    for fid in val_ids:
        # Load full fragment data
        data = data_utils.load_fragment_data(fid, split="val", load_cached_data=True)

        # Run tiled inference
        # Note: We pass the mean/std from the training set for consistent normalization
        prob_map = inference_utils.predict_tiled(
            net,
            data["volume"],
            data["mask"],
            train_dataset.mean,
            train_dataset.std,
            device,
        )

        # Extract valid pixels for metric calculation
        valid_mask = data["mask"] > 0
        flat_p = prob_map[valid_mask]
        flat_l = data["label"][valid_mask]

        all_preds_flat.append(flat_p)
        all_labels_flat.append(flat_l)

        analysis_data.append(
            {
                "fid": fid,
                "volume": data["volume"],
                "label": data["label"],
                "pred": prob_map,
                "mask": data["mask"],
            }
        )

    # Concatenate all pixels from all validation fragments
    if all_preds_flat:
        global_preds = np.concatenate(all_preds_flat)
        global_labels = np.concatenate(all_labels_flat)

        # Optimize Threshold
        best_threshold, best_score = train_utils.optimize_threshold(
            global_preds, global_labels
        )

        # Save threshold for reference
        with open(Config.THRESHOLD_PATH, "w") as f:
            f.write(str(best_threshold))

        print(f"Final Validation Metric: {best_score}")
    else:
        best_score = 0.0
        best_threshold = 0.5
        print("Final Validation Metric: 0.0")

    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate correlation between error magnitude and input intensity
    if analysis_data:
        print("Performing Failure Analysis...")
        # Use the first validation fragment
        item = analysis_data[0]
        vol = item["volume"]
        label = item["label"]
        pred = item["pred"]
        mask = item["mask"]

        # Error map: absolute difference
        error_map = np.abs(pred - label)

        # Input feature: Mean intensity across Z-depth
        # (Using float32 to avoid overflow before mean)
        mean_intensity = np.mean(vol.astype(np.float32), axis=0)

        # Select valid pixels
        valid_indices = mask > 0

        if np.sum(valid_indices) > 0:
            err_vals = error_map[valid_indices]
            int_vals = mean_intensity[valid_indices]

            # Correlation
            if np.std(err_vals) > 0 and np.std(int_vals) > 0:
                corr = np.corrcoef(err_vals, int_vals)[0, 1]
                print(f"Failure Analysis - Error vs Intensity Correlation: {corr}")
            else:
                print(
                    "Failure Analysis - Error vs Intensity Correlation: Undefined (constant values)"
                )

    # 7. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold condition from task description
    TARGET_SCORE = 0.4064630960392697

    if best_score > TARGET_SCORE:
        print("Validation score meets threshold. Generating submission...")

        test_meta_path = Config.METADATA_DIR / "test.csv"
        if test_meta_path.exists():
            test_meta = pd.read_csv(test_meta_path)
            test_ids = test_meta["fragment_id"].astype(str).tolist()

            submission_records = []

            for fid in test_ids:
                # Load test data
                data = data_utils.load_fragment_data(
                    fid, split="test", load_cached_data=True
                )

                # Inference
                prob_map = inference_utils.predict_tiled(
                    net,
                    data["volume"],
                    data["mask"],
                    train_dataset.mean,
                    train_dataset.std,
                    device,
                )

                # Apply Threshold
                binary_map = (prob_map >= best_threshold).astype(np.uint8)

                # Encode
                rle_str = inference_utils.rle_encode(binary_map)
                submission_records.append({"Id": fid, "Predicted": rle_str})

            # Save CSV
            sub_df = pd.DataFrame(submission_records)
            sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print("Test metadata not found. Cannot generate submission.")
    else:
        print(
            f"Validation score ({best_score:.4f}) did not meet target ({TARGET_SCORE:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
