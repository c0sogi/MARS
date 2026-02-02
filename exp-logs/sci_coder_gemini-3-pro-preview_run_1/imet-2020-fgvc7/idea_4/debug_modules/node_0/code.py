import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, ModelEMA
from library.dataset import get_dataloaders
from library.model import ArtworkModel
from library.engine import train_one_epoch, validate, predict


def run_demonstration():
    print("=== Starting Artwork Attribute Labeling Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for rapid demonstration
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch size for demo
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead

    # Set reproducible seed
    seed_everything(Config.SEED)
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading DataLoaders...")
    # debug=True loads a small subset (2000 train, 500 val, 500 test)
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Verification: Check if loaders are populated
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Val loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Verification: Check batch structure
    sample_images, sample_targets = next(iter(train_loader))
    print(
        f"Sample Batch Shape - Images: {sample_images.shape}, Targets: {sample_targets.shape}"
    )

    assert sample_images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image tensor shape."
    assert sample_targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect target tensor shape."
    assert sample_images.dtype == torch.float32, "Images should be float32."
    assert sample_targets.dtype == torch.float32, "Targets should be float32."

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = ArtworkModel(pretrained=True)
    model.to(Config.DEVICE)

    # Verification: Forward pass with dummy input
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch."

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Setting up Optimizer and Training Loop...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize EMA model
    ema_model = ModelEMA(model, decay=Config.EMA_DECAY)

    print("Running training for 1 epoch...")
    train_loss = train_one_epoch(
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        data_loader=train_loader,
        device=Config.DEVICE,
        epoch=1,
    )

    # Verification: Loss validity
    assert not np.isnan(train_loss), "Training loss returned NaN."
    assert train_loss > 0, "Training loss should be positive."
    print(f"Epoch 1 Training Loss: {train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Validation Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Validation...")

    # Use EMA model for validation
    val_loss, val_f1, val_logits, val_targets = validate(
        model=ema_model.module, data_loader=val_loader, device=Config.DEVICE
    )

    # Verification: Metric validity
    assert not np.isnan(val_loss), "Validation loss returned NaN."
    assert 0.0 <= val_f1 <= 1.0, "F1 score must be between 0 and 1."
    assert val_logits.shape == val_targets.shape, "Logits and targets shape mismatch."

    print(f"Validation Results - Loss: {val_loss:.4f}, Micro F1: {val_f1:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    test_logits, test_ids = predict(
        model=ema_model.module, data_loader=test_loader, device=Config.DEVICE
    )

    # Verification: Prediction count
    assert len(test_logits) == len(
        test_ids
    ), "Number of predictions does not match number of IDs."
    print(f"Generated predictions for {len(test_ids)} test images.")

    print("Processing predictions for submission...")
    # Convert logits to probabilities
    probs = torch.sigmoid(test_logits).numpy()

    # Apply threshold (using 0.3 as an example threshold)
    threshold = 0.3
    pred_binary = (probs > threshold).astype(int)

    submission_data = []
    for i, img_id in enumerate(test_ids):
        # Get indices where prediction is 1
        indices = np.where(pred_binary[i] == 1)[0]
        # Format as space-separated string
        indices_str = " ".join(map(str, indices))
        submission_data.append({"id": img_id, "attribute_ids": indices_str})

    submission_df = pd.DataFrame(submission_data)

    # Save submission
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    # Verification: File existence and content
    assert os.path.exists(output_path), "Submission file was not created."
    assert (
        "id" in submission_df.columns and "attribute_ids" in submission_df.columns
    ), "Submission dataframe missing required columns."

    print(f"Submission saved to: {output_path}")
    print("First 3 rows of submission:")
    print(submission_df.head(3))

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
