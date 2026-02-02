import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config, seed_everything
from library.dataset import FractureDataset, get_transforms
from library.model import CervicalFractureNet
from library.loss import HierarchicalCompoundLoss
from library.engine import fit


def main():
    print("=== Cervical Spine Fracture Detection Demo ===")

    # 1. Setup & Configuration Override
    # We override the default Config to ensure the demo runs quickly (within seconds/minutes)
    seed_everything(Config.SEED)

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Limit to 4 samples for speed
    Config.NUM_WORKERS = 0  # Use main thread to avoid multiprocessing overhead in demo

    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG} (Samples: {Config.DEBUG_SAMPLE_SIZE})")

    # 2. Dataset & DataLoader Initialization
    print("\n--- Initializing Datasets ---")

    # Training Dataset
    train_dataset = FractureDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        image_root_dir=Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("train"),
        mode="train",
        debug=Config.DEBUG,
    )

    # Validation Dataset
    val_dataset = FractureDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        image_root_dir=Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("val"),
        mode="val",
        debug=Config.DEBUG,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # Verification: Check Batch Structure
    print("Verifying batch structure...")
    try:
        batch = next(iter(train_loader))
        images = batch["image"]
        targets = batch["targets"]
        patient_targets = batch["patient_target"]

        # Expected Image: (Batch, Seq_Len, Channels, H, W) -> (2, 64, 3, 256, 256)
        assert images.dim() == 5, f"Expected 5D input tensor, got {images.dim()}"
        assert (
            images.shape[1] == Config.NUM_SLICES
        ), f"Expected seq len {Config.NUM_SLICES}, got {images.shape[1]}"
        assert (
            images.shape[2] == 3
        ), f"Expected 3 channels (2.5D), got {images.shape[2]}"

        # Expected Targets: (Batch, 7)
        assert targets.shape == (
            Config.BATCH_SIZE,
            7,
        ), f"Expected targets (B, 7), got {targets.shape}"

        print("Batch structure verified successfully.")
    except Exception as e:
        print(f"Batch verification failed: {e}")
        raise

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = CervicalFractureNet()
    model.to(Config.DEVICE)

    # Verification: Forward Pass
    print("Verifying forward pass...")
    with torch.no_grad():
        dummy_in = images.to(Config.DEVICE)
        dummy_out = model(dummy_in)

    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        7,
    ), f"Expected output (B, 7), got {dummy_out.shape}"
    print("Forward pass verified.")

    # 4. Loss Function Verification
    print("\n--- Initializing Loss ---")
    criterion = HierarchicalCompoundLoss()

    print("Verifying loss calculation...")
    dummy_targets = targets.to(Config.DEVICE)
    dummy_patient = patient_targets.to(Config.DEVICE)

    loss = criterion(dummy_out, dummy_targets, dummy_patient)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive (BCE)"
    print(f"Loss calculation verified: {loss.item():.4f}")

    # 5. Training Loop Execution
    print("\n--- Starting Training Loop (Demo) ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Run the training engine
    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Inference & Submission Generation
    print("\n--- Starting Inference & Submission Generation ---")

    # Initialize Test Dataset
    test_dataset = FractureDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        image_root_dir=Config.TEST_IMAGES_DIR,
        transform=get_transforms("test"),
        mode="test",
        debug=Config.DEBUG,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    trained_model.eval()
    submission_rows = []

    print(f"Processing {len(test_dataset)} test studies...")
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(Config.DEVICE)
            uids = batch["study_uid"]

            # Get logits
            logits = trained_model(imgs)
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()

            # Format predictions for submission
            for i, uid in enumerate(uids):
                # Per-vertebrae probabilities (C1-C7)
                pred = probs[i]

                # Patient overall probability (Max of vertebrae probs)
                p_overall = np.max(pred)

                # Append rows for C1-C7
                for v_idx, v_name in enumerate([f"C{j}" for j in range(1, 8)]):
                    row_id = f"{uid}_{v_name}"
                    submission_rows.append(
                        {"row_id": row_id, "fractured": float(pred[v_idx])}
                    )

                # Append row for patient_overall
                submission_rows.append(
                    {"row_id": f"{uid}_patient_overall", "fractured": float(p_overall)}
                )

    # Save Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verification: Submission File
    print("Verifying submission file...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["row_id", "fractured"]
    if list(loaded_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(loaded_sub.columns)}"
        )

    # Check content
    if len(loaded_sub) == 0:
        raise ValueError("Submission file is empty.")

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print(f"Total rows: {len(loaded_sub)}")
    print("First 5 rows:")
    print(loaded_sub.head().to_string())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
