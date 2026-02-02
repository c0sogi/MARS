import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import library modules
import library.config as C
import library.utils as U
import library.data as D
import library.model as M
import library.train as T


def run_demo():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo
    # -------------------------------------------------------------------------
    # We modify the global configuration to ensure the demo runs quickly
    # and writes to a separate directory.
    print("\n[1] Configuring environment for demo execution...")

    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config paths to point to demo directory
    C.WORKING_DIR = DEMO_DIR
    C.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "roi_cache_train.parquet")
    C.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "roi_cache_val.parquet")
    C.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "roi_cache_test.parquet")
    C.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")

    # Reduce compute load for demo
    C.NUM_EPOCHS = 1
    C.BATCH_SIZE = 2
    C.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    print(f"    Working Directory: {C.WORKING_DIR}")
    print(f"    Batch Size: {C.BATCH_SIZE}")
    print(f"    Epochs: {C.NUM_EPOCHS}")

    # Set seeds for reproducibility
    U.seed_everything(C.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Dataset Logic
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # Load metadata
    if not os.path.exists(C.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {C.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(C.TRAIN_METADATA_PATH)

    # Create a small dataset instance (first 4 samples)
    # We disable cache loading to force raw processing verification
    demo_dataset = D.MGMTDataset(
        metadata_df=train_df,
        transform=D.get_transforms(phase="train"),
        cache_path=None,
        load_cached_data=False,
        debug_limit=4,
    )

    print(f"    Created dataset with {len(demo_dataset)} samples.")

    # Verify item structure
    img, label = demo_dataset[0]

    # Expected shape: (Channels, Height, Width) -> (12, 224, 224)
    expected_shape = (C.INPUT_CHANNELS, C.IMG_SIZE, C.IMG_SIZE)

    print(f"    Sample 0 Shape: {img.shape}")
    print(f"    Sample 0 Label: {label}")

    if img.shape != expected_shape:
        raise AssertionError(
            f"Dataset output shape mismatch. Expected {expected_shape}, got {img.shape}"
        )

    if not isinstance(label, torch.Tensor):
        raise AssertionError("Label is not a torch.Tensor")

    print("    Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = M.AsymmetricEfficientNet()
    model.eval()

    # Create dummy input batch: (Batch_Size, Channels, H, W)
    dummy_input = torch.randn(2, C.INPUT_CHANNELS, C.IMG_SIZE, C.IMG_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Input Shape: {dummy_input.shape}")
    print(f"    Output Shape: {output.shape}")

    # Expected output: (Batch_Size, 1) (Logits)
    if output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
        )

    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Execute Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (via library.train.fit)...")

    # We use the provided fit function.
    # We pass debug_limit to ensure it only processes a tiny subset of data.
    # We disable cache loading to ensure it runs through the processing logic.
    try:
        T.fit(
            epochs=C.NUM_EPOCHS,
            batch_size=C.BATCH_SIZE,
            debug_limit=6,  # Train on 6, Val on 6 (approx, depending on split size)
            load_cached_data=False,
        )
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    if not os.path.exists(C.MODEL_SAVE_PATH):
        # Note: If validation AUC doesn't improve (which is possible in 1 epoch with random weights),
        # the model might not be saved by the checkpoint logic if it strictly requires > best_auc.
        # However, best_auc starts at 0.0. Unless AUC is exactly 0.0, it should save.
        # If it fails, we manually save for the next step.
        print(
            "    Warning: Model checkpoint not found (likely due to short training). Saving current model manually."
        )
        torch.save(model.state_dict(), C.MODEL_SAVE_PATH)
    else:
        print(f"    Training complete. Model saved to {C.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission for Test Set...")

    # Load Test Metadata
    test_df = pd.read_csv(C.TEST_METADATA_PATH)

    # Use a subset for demo speed
    test_df_subset = test_df.head(5).copy()

    # Initialize Test Dataset
    test_dataset = D.MGMTDataset(
        metadata_df=test_df_subset,
        transform=D.get_transforms(phase="test"),  # usually just ToTensor or Normalize
        is_test=True,
        load_cached_data=False,
    )

    test_loader = D.get_dataloader(
        test_dataset, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load Model
    device = torch.device(C.DEVICE)
    model = M.AsymmetricEfficientNet().to(device)
    model.load_state_dict(torch.load(C.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    predictions = []
    ids = []

    print("    Running inference...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)

    # Retrieve IDs from the dataset subset
    # Note: MGMTDataset stores bra_ids in self.bra_ids
    ids = test_dataset.bra_ids

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Save
    sub_path = os.path.join(DEMO_DIR, "submission_demo.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"    Submission saved to {sub_path}")
    print("    Head:")
    print(submission_df.head())

    # Verify format
    if list(submission_df.columns) != ["BraTS21ID", "MGMT_value"]:
        raise AssertionError("Submission columns mismatch.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
