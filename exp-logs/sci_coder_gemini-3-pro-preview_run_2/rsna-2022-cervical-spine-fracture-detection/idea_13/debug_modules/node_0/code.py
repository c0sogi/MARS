import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import CervicalSpineDataset, get_transforms
from library.model import CervicalFractureNet
from library.loss import HybridLoss
from library.engine import train_one_epoch, validate


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- 1. Configuration & Setup ---")

    # Set reproducibility
    Config.setup_reproducibility(seed=42)

    # Override Config parameters for a fast demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.GRAD_ACCUM_STEPS = 1
    Config.SEQ_LEN = 32  # Reduced from 96 to speed up demo
    Config.NUM_WORKERS = 2

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    print("\n--- 2. Data Loading ---")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create small subsets for demonstration (4 train samples, 2 val samples)
    train_subset = train_df.iloc[:4].reset_index(drop=True)
    val_subset = val_df.iloc[:2].reset_index(drop=True)

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    # Initialize Datasets
    train_dataset = CervicalSpineDataset(
        train_subset, mode="train", transform=get_transforms("train")
    )
    val_dataset = CervicalSpineDataset(
        val_subset, mode="val", transform=get_transforms("val")
    )

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    images = sample_batch["images"]
    targets = sample_batch["targets"]

    print(f"Sample Batch - Images Shape: {images.shape}")
    print(f"Sample Batch - Targets Shape: {targets.shape}")

    # Assertions to verify data pipeline
    expected_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert (
        images.shape == expected_shape
    ), f"Expected {expected_shape}, got {images.shape}"
    assert targets.shape == (Config.BATCH_SIZE, 8), "Targets shape mismatch"

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- 3. Model Initialization ---")

    model = CervicalFractureNet()
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        output = model(dummy_input)

        logits = output["logits"]
        attn_weights = output["attn_weights"]

        print(f"Output Logits Shape: {logits.shape}")
        print(f"Output Attn Weights Shape: {attn_weights.shape}")

        assert logits.shape == (Config.BATCH_SIZE, 8), "Logits shape mismatch"
        assert attn_weights.shape == (
            Config.BATCH_SIZE,
            8,
            Config.SEQ_LEN,
        ), "Attention weights shape mismatch"

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n--- 4. Training Loop ---")

    loss_fn = HybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run for defined epochs
    for epoch in range(1, Config.EPOCHS + 1):
        epoch_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=None,  # No scheduler for short demo
            dataloader=train_loader,
            device=device,
            loss_fn=loss_fn,
            epoch=epoch,
        )

        assert not np.isnan(epoch_loss), "Training loss resulted in NaN"

    # ==========================================
    # 5. Validation & Metrics
    # ==========================================
    print("\n--- 5. Validation ---")

    val_loss, metric_score, val_preds = validate(
        model=model, dataloader=val_loader, device=device, loss_fn=loss_fn
    )

    print(f"Final Validation Metric: {metric_score:.6f}")

    # Verify predictions dataframe structure
    assert "row_id" in val_preds.columns
    assert "fractured" in val_preds.columns
    # 2 studies * 8 classes = 16 rows
    assert len(val_preds) == len(val_subset) * 8, "Prediction row count mismatch"

    # ==========================================
    # 6. Inference (Test Set)
    # ==========================================
    print("\n--- 6. Inference Demo ---")

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_subset = test_df.iloc[:2].reset_index(drop=True)  # Subset for speed

    test_dataset = CervicalSpineDataset(
        test_subset,
        mode="test",
        transform=get_transforms("val"),  # Use validation transforms (deterministic)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    model.eval()
    submission_rows = []
    class_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["images"].to(device)
            uids = batch["row_id"]

            # Forward pass
            out = model(imgs)
            logits = out["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()

            # Format predictions
            for i, uid in enumerate(uids):
                for c_idx, c_name in enumerate(class_names):
                    submission_rows.append(
                        {
                            "row_id": f"{uid}_{c_name}",
                            "fractured": float(probs[i, c_idx]),
                        }
                    )

    # Save Submission
    submission_df = pd.DataFrame(submission_rows)
    output_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(f"Total submission rows: {len(submission_df)}")
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
