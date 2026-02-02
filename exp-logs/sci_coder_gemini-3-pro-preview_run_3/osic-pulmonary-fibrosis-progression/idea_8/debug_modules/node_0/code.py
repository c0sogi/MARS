import os
import sys
import torch
import pandas as pd
import numpy as np
from library import config, utils, preprocess, data, model, engine


def main():
    # 1. Setup
    print("=== Setting up environment ===")
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Preparation (Simulate Preprocessing)
    print("\n=== Preparing Data Subset ===")

    # Load metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Identify the subset of patients used when debug=True in data.py
    # get_train_val_loaders(debug=True) slices the first 50 train and 20 val rows
    debug_train_patients = train_df.iloc[:50]["Patient"].unique()
    debug_val_patients = val_df.iloc[:20]["Patient"].unique()
    test_patients = test_df["Patient"].unique()

    # Combine all needed patients
    patients_to_process = (
        set(debug_train_patients) | set(debug_val_patients) | set(test_patients)
    )
    print(f"Selected {len(patients_to_process)} patients for processing.")

    # Generate auxiliary slopes file (Required by LungDataset)
    # We compute this on the full combined train/val set as per preprocess.py logic
    print("Generating patient slopes...")
    full_history_df = pd.concat([train_df, val_df], ignore_index=True)
    slope_df = preprocess.compute_patient_slopes(full_history_df)
    slope_path = os.path.join(config.WORKING_DIR, "patient_slopes.csv")
    slope_df.to_csv(slope_path, index=False)

    # Process Images for the selected subset
    # We manually call process_patient to avoid running on the entire dataset
    print("Processing images (caching to working directory)...")
    processed_count = 0
    for pid in patients_to_process:
        save_path = os.path.join(config.CACHE_DIR, f"{pid}.npy")

        # Skip if already cached
        if os.path.exists(save_path):
            continue

        # Locate source directory
        train_dir = os.path.join(config.INPUT_DIR, "train", pid)
        test_dir = os.path.join(config.INPUT_DIR, "test", pid)

        if os.path.exists(train_dir):
            preprocess.process_patient(
                pid, os.path.join(config.INPUT_DIR, "train"), save_path
            )
            processed_count += 1
        elif os.path.exists(test_dir):
            preprocess.process_patient(
                pid, os.path.join(config.INPUT_DIR, "test"), save_path
            )
            processed_count += 1

    print(f"Processed {processed_count} new images.")

    # 3. Data Loading
    print("\n=== Initializing DataLoaders ===")
    # debug=True ensures we only load the subset of data we prepared
    train_loader, val_loader = data.get_train_val_loaders(debug=True)

    # Verify Batch Structure
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Tabular shape: {batch['tabular'].shape}")

    assert batch["image"].shape[1:] == (
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), "Incorrect image dimensions"
    assert batch["tabular"].shape[1] == 7, "Incorrect tabular dimensions"

    # 4. Model Initialization
    print("\n=== Initializing Model & Engine ===")
    net = model.TAPNet().to(device)

    # Setup Optimizer and Scheduler
    optimizer = engine.get_optimizer(net)
    scheduler = engine.get_scheduler(optimizer, epochs=1)

    # Initialize Engine
    tap_engine = engine.TAPNetEngine(net, device, optimizer, scheduler)
    print("Model initialized successfully.")

    # 5. Training Loop
    print("\n=== Starting Training (1 Epoch) ===")
    train_metrics = tap_engine.train_one_epoch(train_loader)
    print(f"Train Metrics: {train_metrics}")

    assert not pd.isna(train_metrics["loss"]), "Training loss is NaN"
    assert not pd.isna(train_metrics["nll"]), "NLL loss is NaN"

    # 6. Evaluation Loop
    print("\n=== Starting Evaluation ===")
    val_metrics = tap_engine.evaluate(val_loader)
    print(f"Validation Metrics: {val_metrics}")

    # Metric is negative and higher is better (e.g., -6.5 is better than -7.0)
    assert val_metrics["score"] < 0, "Metric score should be negative"

    # 7. Inference / Submission
    print("\n=== Running Inference ===")

    # Prepare a demo submission file
    # We filter the sample_submission.csv to only include patients present in our test set
    # and limit rows for speed.
    sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)

    # Extract Patient ID from Patient_Week column
    sample_sub["Patient_ID"] = sample_sub["Patient_Week"].apply(
        lambda x: x.split("_")[0]
    )

    # Filter for patients that actually exist in the test set metadata
    valid_test_patients = set(test_df["Patient"].unique())
    valid_sub = sample_sub[sample_sub["Patient_ID"].isin(valid_test_patients)].copy()

    # Take a small subset for demonstration
    demo_sub = valid_sub.iloc[:50].drop(columns=["Patient_ID"])
    demo_sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    demo_sub.to_csv(demo_sub_path, index=False)

    # Get Loader
    test_loader = data.get_submission_loader(demo_sub_path)

    # Run Prediction Loop
    net.eval()
    predictions = []

    print("Predicting...")
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)

            # Forward Pass
            params = net(imgs, tabs)

            # Calculate Trajectory (mu, sigma) in scaled space
            mu_scaled, sigma_scaled = tap_engine.calculate_trajectory(
                params, tabs, weeks
            )

            # Unscale to original units (ml)
            fvc_pred = mu_scaled * config.STATS["FVC_STD"] + config.STATS["FVC_MEAN"]
            sigma_pred = sigma_scaled * config.STATS["FVC_STD"]

            # Collect results
            p_ids = batch["patient_id"]
            raw_weeks = batch["raw_weeks"]

            for i in range(len(p_ids)):
                predictions.append(
                    {
                        "Patient_Week": f"{p_ids[i]}_{raw_weeks[i].item()}",
                        "FVC": fvc_pred[i].item(),
                        "Confidence": sigma_pred[i].item(),
                    }
                )

    # Create Submission DataFrame
    pred_df = pd.DataFrame(predictions)
    print("\nSample Predictions:")
    print(pred_df.head())

    assert len(pred_df) == 50, "Prediction count mismatch"
    assert not pred_df.isnull().values.any(), "Predictions contain NaNs"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
