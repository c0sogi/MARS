import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import CervicalFractureNet
from library.loss import HybridLoss
from library.train import Trainer
from library.utils import seed_everything, calculate_weighted_log_loss


def main():
    # 1. Setup and Configuration
    print(">>> Setting up configuration for demo...")
    seed_everything(42)
    warnings.filterwarnings("ignore")

    # Instantiate and override Config for speed and resource efficiency
    config = Config()

    # Enable Debug mode to use a tiny subset of data
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 12  # Small sample size for demo

    # Reduce dimensions for speed
    config.IMAGE_SIZE = (128, 128)
    config.SEQ_LENGTH = 16  # Shorter sequence
    config.BATCH_SIZE = 2  # Small batch

    # Training settings for a quick run
    config.EPOCHS = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    config.WORKING_DIR = "./working/demo_execution"
    config.MODEL_SAVE_PATH = os.path.join(config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"    Working Directory: {config.WORKING_DIR}")
    print(f"    Image Size: {config.IMAGE_SIZE}")
    print(f"    Sequence Length: {config.SEQ_LENGTH}")

    # 2. Data Loading Demonstration
    print("\n>>> Demonstrating Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["images"]
    fracture_labels = batch["fracture_labels"]
    aux_labels = batch["aux_labels"]
    study_ids = batch["study_id"]

    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Images shape: {images.shape} (Expected: [B, Seq, C, H, W])")
    print(
        f"    Fracture Labels shape: {fracture_labels.shape} (Expected: [B, Num_Classes])"
    )
    print(f"    Aux Labels shape: {aux_labels.shape} (Expected: [B, Seq])")

    # Verification
    assert images.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        3,
        128,
        128,
    ), "Image shape mismatch"
    assert fracture_labels.shape == (
        config.BATCH_SIZE,
        8,
    ), "Fracture label shape mismatch"
    assert aux_labels.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
    ), "Aux label shape mismatch"

    # 3. Model Forward Pass Demonstration
    print("\n>>> Demonstrating Model Forward Pass...")
    model = CervicalFractureNet(config)
    model.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)
    fracture_labels = fracture_labels.to(config.DEVICE)
    aux_labels = aux_labels.to(config.DEVICE)

    # Forward pass
    outputs = model(images)
    fracture_logits = outputs["fracture_logits"]
    aux_logits = outputs["aux_logits"]

    print(f"    Fracture Logits shape: {fracture_logits.shape}")
    print(f"    Aux Logits shape: {aux_logits.shape}")

    # Verification
    assert fracture_logits.shape == (
        config.BATCH_SIZE,
        8,
    ), "Output fracture logits shape mismatch"
    assert aux_logits.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        8,
    ), "Output aux logits shape mismatch"

    # 4. Loss Calculation Demonstration
    print("\n>>> Demonstrating Loss Calculation...")
    criterion = HybridLoss()

    targets = {"fracture_labels": fracture_labels, "aux_labels": aux_labels}

    loss_dict = criterion(outputs, targets)

    print(f"    Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"    Main Loss: {loss_dict['main_loss'].item():.4f}")
    print(f"    Aux Loss: {loss_dict['aux_loss'].item():.4f}")

    # Verification
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
    assert loss_dict["loss"] > 0, "Loss should be positive"

    # 5. Training Loop Demonstration
    print("\n>>> Demonstrating Training Loop (1 Epoch)...")
    # Initialize Trainer with our modified config
    trainer = Trainer(config)

    # Run training
    trainer.fit()

    # Verification
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved"
    print(f"    Training complete. Model saved to {config.MODEL_SAVE_PATH}")

    # 6. Metric Calculation Demonstration
    print("\n>>> Demonstrating Metric Calculation...")

    # Create dummy Ground Truth DataFrame
    # Columns: StudyInstanceUID, patient_overall, C1..C7
    y_true_data = {
        "StudyInstanceUID": ["1.2.3.4", "1.2.3.5"],
        "patient_overall": [1, 0],
        "C1": [0, 0],
        "C2": [1, 0],
        "C3": [0, 0],
        "C4": [0, 0],
        "C5": [0, 0],
        "C6": [0, 0],
        "C7": [0, 0],
    }
    y_true_df = pd.DataFrame(y_true_data)

    # Create dummy Prediction DataFrame
    # Rows: 8 per study (Total 16)
    row_ids = []
    fractured_probs = []

    # Study 1 (Fractured C2) - Perfect prediction
    row_ids.extend(
        [
            "1.2.3.4_patient_overall",
            "1.2.3.4_C1",
            "1.2.3.4_C2",
            "1.2.3.4_C3",
            "1.2.3.4_C4",
            "1.2.3.4_C5",
            "1.2.3.4_C6",
            "1.2.3.4_C7",
        ]
    )
    fractured_probs.extend([0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01])

    # Study 2 (Healthy) - Perfect prediction
    row_ids.extend(
        [
            "1.2.3.5_patient_overall",
            "1.2.3.5_C1",
            "1.2.3.5_C2",
            "1.2.3.5_C3",
            "1.2.3.5_C4",
            "1.2.3.5_C5",
            "1.2.3.5_C6",
            "1.2.3.5_C7",
        ]
    )
    fractured_probs.extend([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    y_pred_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured_probs})

    # Calculate score
    score = calculate_weighted_log_loss(y_true_df, y_pred_df)
    print(f"    Calculated Score (Low loss expected): {score:.6f}")

    # Verification
    assert isinstance(score, float), "Score is not a float"
    assert score < 0.1, "Score should be low for near-perfect predictions"

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
