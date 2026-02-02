import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Import Library Components ---
from library.config import Config, seed_everything
from library.dataset import CervicalSpineDataset
from library.model import AnatomicallyAwareModel
from library.loss import WeightedFractureLoss
from library.train import run_training
from library.utils import load_checkpoint


def main():
    print("=== Starting Cervical Spine Fracture Detection Demo ===\n")

    # --- 2. Patch Configuration for Speed and Demo Purposes ---
    print("[1/6] Patching Configuration...")

    # Use a writable directory for this demo execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Modify Hyperparameters for fast execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Use only 6 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.SEQ_LEN = 8  # Reduced sequence length (z-depth)
    Config.IMAGE_SIZE = (128, 128)  # Reduced image resolution
    Config.BACKBONE = "resnet18"  # Lightweight backbone for demo
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.GRAD_ACCUM_STEPS = 1  # No accumulation needed for small batch

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration patched successfully.")

    # --- 3. Dataset Instantiation and Verification ---
    print("\n[2/6] Verifying Dataset...")

    # Initialize Training Dataset
    train_dataset = CervicalSpineDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        phase="train",
        load_cached_data=False,  # Force re-compute for demo
    )

    print(f"Dataset initialized. Size: {len(train_dataset)}")

    # Validate Dataset Item
    sample_idx = 0
    images, targets = train_dataset[sample_idx]

    print(f"Sample {sample_idx} - Image Shape: {images.shape}")
    print(f"Sample {sample_idx} - Target Shape: {targets.shape}")

    # Assertions
    # Shape: (Seq_Len, Channels, H, W) -> (8, 3, 128, 128)
    expected_shape = (
        Config.SEQ_LEN,
        Config.IN_CHANS,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert (
        images.shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {images.shape}"

    # Targets: (8,) -> [C1...C7, patient_overall]
    assert targets.shape == (
        Config.NUM_CLASSES,
    ), f"Expected target shape ({Config.NUM_CLASSES},), got {targets.shape}"

    print("Dataset verification passed.")

    # --- 4. Model Initialization and Forward Pass ---
    print("\n[3/6] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = AnatomicallyAwareModel().to(device)

    # Create a batch of size 2
    batch_images = torch.stack([images, images]).to(
        device
    )  # Shape: (2, 8, 3, 128, 128)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(batch_images)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected logits shape (2, {Config.NUM_CLASSES}), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model produced NaN values."

    print("Model forward pass passed.")

    # --- 5. Loss Function Verification ---
    print("\n[4/6] Verifying Loss Function...")

    loss_fn = WeightedFractureLoss().to(device)
    batch_targets = torch.stack([targets, targets]).to(device)

    # Calculate Loss
    loss = loss_fn(logits, batch_targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert loss.item() > 0, "Loss should be positive."
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."

    print("Loss function verification passed.")

    # --- 6. Full Training Loop Execution ---
    print("\n[5/6] Executing Training Loop (1 Epoch)...")

    # This calls the provided library function which handles loops, logging, and saving
    run_training()

    # Verify Checkpoint Creation
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created."
    print(f"Training finished. Checkpoint saved at {Config.CHECKPOINT_PATH}")

    # --- 7. Inference and Submission Generation ---
    print("\n[6/6] Simulating Inference and Submission...")

    # Load Test Metadata
    # Note: The test metadata provided in the environment usually covers all studies.
    # We will use the provided test_metadata.csv
    test_dataset = CervicalSpineDataset(
        metadata_path=Config.TEST_METADATA_PATH, phase="test", load_cached_data=False
    )

    # Limit test set for demo speed
    if len(test_dataset) > 5:
        test_dataset.df = test_dataset.df.head(5)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Load Best Model
    checkpoint = load_checkpoint(model, Config.CHECKPOINT_PATH)
    model.eval()

    predictions = []

    print("Running inference on test subset...")
    with torch.no_grad():
        for images, study_uid_tuple in test_loader:
            images = images.to(device)
            study_uid = study_uid_tuple[0]

            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # Shape (8,)

            # Map probabilities to submission format
            # Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
            target_cols = Config.TARGET_COLS

            for i, col in enumerate(target_cols):
                row_id = f"{study_uid}_{col}"
                predictions.append({"row_id": row_id, "fractured": probs[i]})

    # Create Submission DataFrame
    submission_df = pd.DataFrame(predictions)

    # Save Submission
    output_path = os.path.join(Config.WORKING_DIR, "output", "submission.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission generated with {len(submission_df)} rows.")
    print(f"Saved to: {output_path}")
    print("\nFirst 5 rows of submission:")
    print(submission_df.head().to_markdown(index=False))

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
