import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library
from library import config, utils, data, model, training, inference


def main():
    print("=== Starting Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # ---------------------------------------------------------
    print("\n[Step 1] Overriding Configuration for Demo...")

    # Set deterministic seed
    utils.set_seed(42)

    # Modify config to use a temporary working directory
    config.WORKING_DIR = os.path.join(config.BASE_DIR, "working", "demo_execution")
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Limit dataset size for speed
    config.DEBUG_MAX_SAMPLES = 32

    # Adjust training parameters for a quick run
    config.BATCH_SIZE = 8
    config.EPOCHS = 2
    config.SWA_START_EPOCH = 0  # Trigger SWA immediately to test that logic
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple demo

    # Reduce TTA widths to a single width for faster inference demo
    # (In a real run, this would be [608, 640, 672])
    config.TTA_WIDTHS = [640]

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Max Samples: {config.DEBUG_MAX_SAMPLES}")
    print(f"Device: {config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("\n[Step 2] Demonstrating Data Loading...")

    # Get DataLoaders
    train_loader, val_loader, test_loader = data.get_dataloaders(
        use_pseudo_labels=False,
        load_cached_data=False,  # Force reload to ensure logic runs
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch from training to verify DynamicBatchCollate
    images, labels = next(iter(train_loader))

    print(f"Sample Batch Shape - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images should be 4D (B, C, H, W)"
    assert labels.dim() == 2, "Labels should be 2D (B, NumClasses)"
    assert (
        labels.shape[1] == config.NUM_CLASSES
    ), f"Labels should have {config.NUM_CLASSES} classes"
    assert images.shape[2] == config.IMG_HEIGHT, f"Height should be {config.IMG_HEIGHT}"
    # Width is dynamic, just checking it exists
    assert images.shape[3] > 0

    print("Data loading verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[Step 3] Demonstrating Model Initialization...")

    net = model.get_model(device=config.DEVICE)

    # Verify forward pass
    dummy_input = images.to(config.DEVICE)
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        images.shape[0],
        config.NUM_CLASSES,
    ), "Output shape mismatch"

    print("Model initialized and verified.")

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[Step 4] Demonstrating Training Loop (Trainer)...")

    # Setup Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Initialize Trainer
    trainer = training.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # Skipping scheduler for short demo
        device=config.DEVICE,
        checkpoint_dir=config.WORKING_DIR,
    )

    # Run Training
    # We set EPOCHS=2 and SWA_START_EPOCH=0, so it should do SWA logic immediately
    trainer.fit(epochs=config.EPOCHS)

    # Verify Checkpoints
    expected_files = ["model_last.pth", "model_best.pth", "model_swa.pth"]
    for fname in expected_files:
        fpath = os.path.join(config.WORKING_DIR, fname)
        if os.path.exists(fpath):
            print(f"Verified checkpoint exists: {fname}")
        else:
            raise FileNotFoundError(f"Checkpoint {fname} was not created!")

    print("Training loop completed successfully.")

    # ---------------------------------------------------------
    # 5. Inference & TTA Demonstration
    # ---------------------------------------------------------
    print("\n[Step 5] Demonstrating Inference and TTA...")

    # Standard Inference on Validation Set
    val_probs = inference.predict_probs(net, val_loader, config.DEVICE)
    print(f"Validation Predictions Shape: {val_probs.shape}")
    assert val_probs.shape[1] == config.NUM_CLASSES

    # TTA Inference on Test Set
    # Note: predict_tta internally loads the test loader based on TTA_WIDTHS
    # Since we set DEBUG_MAX_SAMPLES, the test set size is limited
    tta_probs = inference.predict_tta(net, config.DEVICE)
    print(f"TTA Predictions Shape: {tta_probs.shape}")

    # Verify TTA output size matches the (limited) test set size
    # We loaded metadata earlier via get_dataloaders, let's check size
    test_df = data.load_metadata("test", load_cached_data=True)
    expected_rows = min(len(test_df), config.DEBUG_MAX_SAMPLES)
    assert (
        tta_probs.shape[0] == expected_rows
    ), f"Expected {expected_rows} predictions, got {tta_probs.shape[0]}"

    print("Inference verified.")

    # ---------------------------------------------------------
    # 6. Submission & Pseudo-Label Generation
    # ---------------------------------------------------------
    print("\n[Step 6] Demonstrating Submission and Pseudo-Label Generation...")

    # Generate Submission File
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    inference.generate_submission(net, config.DEVICE, submission_path)

    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created with {len(df_sub)} rows.")
        # Check columns
        assert "Id" in df_sub.columns and "Probability" in df_sub.columns
        # Check row count: (N_samples * N_classes)
        assert len(df_sub) == expected_rows * config.NUM_CLASSES
    else:
        raise FileNotFoundError("Submission file not created.")

    # Save Pseudo Labels
    pseudo_path = os.path.join(config.WORKING_DIR, "demo_pseudo_labels.parquet")
    inference.save_pseudo_labels(tta_probs, pseudo_path)

    if os.path.exists(pseudo_path):
        df_pseudo = pd.read_parquet(pseudo_path)
        print(f"Pseudo-labels file created with shape: {df_pseudo.shape}")
        assert "rec_id" in df_pseudo.columns
        assert len(df_pseudo) == expected_rows
    else:
        raise FileNotFoundError("Pseudo-labels file not created.")

    print("\n=== Demonstration Complete: All components verified successfully ===")


if __name__ == "__main__":
    main()
