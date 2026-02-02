import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import get_dataloader
from library.model import CatheterModel
from library.utils import (
    calculate_pos_weights,
    train_one_epoch,
    validate,
    generate_submission,
)


def main():
    print("Starting Catheter Detection Pipeline Demo...")

    # 1. Setup and Reproducibility
    seed_everything(42)

    # Override Config for speed and demonstration purposes
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG = True

    # We will use a small subset of data for this demo
    DEMO_SIZE = 32

    print(
        f"Configuration: Device={Config.DEVICE}, Batch Size={Config.BATCH_SIZE}, Subset Size={DEMO_SIZE}"
    )

    # 2. Data Loading Demonstration
    print("\n--- Data Loading ---")

    # Initialize DataLoaders for Train, Valid, and Test
    # We use debug_size to limit the number of samples loaded
    train_loader = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, debug_size=DEMO_SIZE
    )
    val_loader = get_dataloader(
        "valid", batch_size=Config.BATCH_SIZE, debug_size=DEMO_SIZE
    )
    test_loader = get_dataloader(
        "test", batch_size=Config.BATCH_SIZE, debug_size=DEMO_SIZE
    )

    print("DataLoaders initialized.")

    # Fetch a single batch to verify shapes and types
    images, labels = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Labels Shape: {labels.shape}")

    # Assertions for Data Loading
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected label shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"
    print("Data loading verification passed.")

    # 3. Model Initialization Demonstration
    print("\n--- Model Initialization ---")

    # Initialize model
    # We use pretrained=False to avoid downloading weights during this quick demo
    model = CatheterModel(pretrained=False)
    model.to(Config.DEVICE)

    print("Model initialized and moved to device.")

    # Verify Forward Pass
    with torch.no_grad():
        # Move sample batch to device
        sample_input = images.to(Config.DEVICE)
        outputs = model(sample_input)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for Model
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"
    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- Training Step ---")

    # Calculate positive weights for loss function
    # Note: This might read the full metadata to calc weights, but it's cached or fast enough
    pos_weights = calculate_pos_weights(load_cached_data=False, device=Config.DEVICE)
    print(f"Positive weights calculated. Shape: {pos_weights.shape}")

    # Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch (on the small subset)
    print("Executing training for one epoch...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, Config.DEVICE, pos_weights
    )

    print(f"Training Loss: {train_loss:.4f}")

    # Assertions for Training
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print("Training step verification passed.")

    # 5. Validation Demonstration
    print("\n--- Validation Step ---")

    print("Executing validation...")
    avg_auc, auc_scores = validate(model, val_loader, Config.DEVICE)

    print(f"Validation Average AUC: {avg_auc:.4f}")

    # Assertions for Validation
    assert isinstance(avg_auc, float), "Average AUC should be a float"
    assert isinstance(auc_scores, dict), "AUC scores should be a dictionary"
    assert (
        len(auc_scores) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} AUC scores, got {len(auc_scores)}"
    print("Validation step verification passed.")

    # 6. Submission Generation Demonstration
    print("\n--- Submission Generation ---")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    print(f"Generating predictions for test set (subset size: {DEMO_SIZE})...")
    generate_submission(model, test_loader, Config.DEVICE, output_path=submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Assertions for Submission
    expected_cols = ["StudyInstanceUID"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match requirements"

    # Since we used debug_size=DEMO_SIZE, the submission should have DEMO_SIZE rows
    # (assuming the test set is larger than DEMO_SIZE, which it is)
    assert (
        len(df_sub) == DEMO_SIZE
    ), f"Expected {DEMO_SIZE} rows in submission, got {len(df_sub)}"

    # Check probabilities range
    pred_cols = Config.TARGET_COLS
    preds = df_sub[pred_cols].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]"

    print("Submission verification passed.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
