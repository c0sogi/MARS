import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys

# Import the provided library modules
from library import config, utils, data, model


def run_demonstration():
    print(">>> Starting MGMT Methylation Prediction Pipeline Demonstration")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Ensure reproducibility
    config.seed_everything(42)

    # Override batch size for this demo to work with small data subsets
    config.BATCH_SIZE = 2

    # Define a demo working directory
    DEMO_DIR = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Working Directory: {DEMO_DIR}")
    print(f"Device: {config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Processing (library.utils)
    # -------------------------------------------------------------------------
    print("\n[1/5] Processing Data Subset...")

    # Load the training metadata
    if not os.path.exists(config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA}")

    df_full = pd.read_csv(config.TRAIN_METADATA)

    # Select a tiny subset (4 patients) for speed
    df_subset = df_full.head(4).copy()
    print(f"Selected {len(df_subset)} patients for demonstration.")

    X_list = []
    y_list = []

    # Manually process patients using the library utility
    for idx, row in df_subset.iterrows():
        # utils.process_patient reads DICOMs, handles the Safe Zone/Anchor logic,
        # and returns a (12, 224, 224) tensor and a label.
        X_subj, y_subj = utils.process_patient(row)

        # Validation
        if X_subj.shape != (12, 224, 224):
            raise ValueError(f"Incorrect shape from process_patient: {X_subj.shape}")

        X_list.append(X_subj)
        y_list.append(y_subj)

    X_demo = np.array(X_list, dtype=np.float32)
    y_demo = np.array(y_list, dtype=np.float32)

    print(f"Processed Data Shapes -> X: {X_demo.shape}, y: {y_demo.shape}")

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader (library.data)
    # -------------------------------------------------------------------------
    print("\n[2/5] Creating Datasets and Loaders...")

    # Instantiate the custom Dataset class
    train_dataset = data.MGMTDataset(X_demo, y_demo, augment=True)
    val_dataset = data.MGMTDataset(X_demo, y_demo, augment=False)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,  # False for demo to ensure we get all data
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Verify DataLoader output
    batch_X, batch_y = next(iter(train_loader))
    print(f"Batch X shape: {batch_X.shape}")  # Should be (2, 12, 224, 224)
    print(f"Batch y shape: {batch_y.shape}")  # Should be (2,)

    # -------------------------------------------------------------------------
    # 4. Model Initialization (library.model)
    # -------------------------------------------------------------------------
    print("\n[3/5] Initializing Asymmetric EfficientNet...")

    net = model.AsymmetricEfficientNet().to(config.DEVICE)

    # Verify the architectural modification (Groups=4 in first layer)
    first_layer = net.backbone.features[0][0]
    print(f"First Conv Layer Configuration: {first_layer}")

    if first_layer.groups != 4:
        raise AssertionError(
            "Model stem does not have groups=4 for modality isolation."
        )
    if first_layer.in_channels != 12:
        raise AssertionError(
            "Model input channels should be 12 (4 modalities * 3 slices)."
        )

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = net(batch_X.to(config.DEVICE))
        print(f"Forward pass output shape: {dummy_out.shape}")
        if dummy_out.shape != (config.BATCH_SIZE, 1):
            raise AssertionError(
                f"Expected output shape ({config.BATCH_SIZE}, 1), got {dummy_out.shape}"
            )

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution (library.model functions)
    # -------------------------------------------------------------------------
    print("\n[4/5] Executing Training Step...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Run one epoch of training
    train_loss, train_auc = model.train_one_epoch(
        net, train_loader, criterion, optimizer, config.DEVICE
    )
    print(f"Train Result -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    # Run validation
    val_loss, val_auc = model.validate(net, val_loader, criterion, config.DEVICE)
    print(f"Val Result   -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Save the model (simulating checkpointing)
    best_model_path = os.path.join(DEMO_DIR, "best_model.pth")
    torch.save(net.state_dict(), best_model_path)
    print(f"Model checkpoint saved to: {best_model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission (library.model.predict_and_submit)
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Submission...")

    # To demonstrate predict_and_submit, we need a test metadata file that matches
    # the number of samples in our loader. We'll create a dummy one based on our subset.
    dummy_test_csv = os.path.join(DEMO_DIR, "demo_test_metadata.csv")
    df_test_dummy = df_subset.drop(columns=["MGMT_value"]).copy()
    df_test_dummy.to_csv(dummy_test_csv, index=False)

    # Temporarily override the config path so predict_and_submit reads our dummy file
    original_test_metadata_path = config.TEST_METADATA
    config.TEST_METADATA = dummy_test_csv

    try:
        # Use the val_loader as a proxy for test data
        submission_output_path = os.path.join(DEMO_DIR, "demo_submission.csv")

        model.predict_and_submit(
            net, val_loader, config.DEVICE, output_path=submission_output_path
        )

        # Verify the submission file
        if os.path.exists(submission_output_path):
            sub_df = pd.read_csv(submission_output_path)
            print("\nGenerated Submission File Head:")
            print(sub_df.head())

            if len(sub_df) != len(df_subset):
                raise AssertionError(
                    f"Submission length {len(sub_df)} mismatch with input {len(df_subset)}"
                )
            if "BraTS21ID" not in sub_df.columns or "MGMT_value" not in sub_df.columns:
                raise AssertionError("Submission columns missing.")
        else:
            raise FileNotFoundError("Submission file was not created.")

    finally:
        # Restore configuration
        config.TEST_METADATA = original_test_metadata_path

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    run_demonstration()
