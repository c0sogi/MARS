import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_test_loader, get_mixup_fn
from library.model import get_model, ModelEMA
from library.engine import train_one_epoch, evaluate, inference


def main():
    print("Starting Cassava Leaf Disease Classification Demo...")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for a fast demonstration run
    Config.DEBUG = True  # Uses a tiny subset of data (100 train, 50 val)
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.P1_EPOCHS = 1  # Run only 1 epoch
    Config.P1_BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers

    # Initialize directories
    Config.setup(debug=True)

    print(
        f"Configuration set. Device: {Config.DEVICE}, Output Dir: {Config.OUTPUT_DIR}"
    )

    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 2. Data Loading Demonstration ---")

    # Get DataLoaders for Fold 0
    # In DEBUG mode, this returns a very small subset
    train_loader, val_loader = get_dataloaders(
        fold_idx=0,
        img_size=224,
        batch_size=Config.P1_BATCH_SIZE,
        load_cached_data=False,  # Force regeneration for demo
    )

    # Verify Train Loader
    print(f"Train Loader Length: {len(train_loader)}")
    assert len(train_loader) > 0, "Train loader is empty."

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Should be (8, 3, 224, 224)
    print(f"Batch Label Shape: {labels.shape}")  # Should be (8,)

    assert images.shape == (
        Config.P1_BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect image batch shape."
    assert labels.shape == (Config.P1_BATCH_SIZE,), "Incorrect label batch shape."

    # Verify Mixup Function
    mixup_fn = get_mixup_fn(mixup_prob=0.5, label_smoothing=0.1)
    if mixup_fn:
        mixed_images, mixed_labels = mixup_fn(images, labels)
        print(
            f"Mixed Label Shape: {mixed_labels.shape}"
        )  # Should be (8, 5) due to one-hot/smoothing
        assert mixed_labels.shape == (
            Config.P1_BATCH_SIZE,
            Config.NUM_CLASSES,
        ), "Mixup labels incorrect shape."

    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n--- 3. Model Instantiation ---")

    # Initialize Model (pretrained=False for speed/offline safety in demo)
    model = get_model(pretrained=False)
    print(f"Model {Config.MODEL_NAME} initialized.")

    # Initialize EMA
    model_ema = ModelEMA(model)
    print("ModelEMA initialized.")

    # Verify device placement
    assert (
        next(model.parameters()).device.type == Config.DEVICE
    ), "Model not on correct device."

    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 4. Training Loop Demonstration ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train for one epoch
    # Note: In DEBUG mode, we only have ~100 samples, so this is very fast
    train_loss, train_acc = train_one_epoch(
        epoch=0,
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        device=Config.DEVICE,
        scheduler=None,
        mixup_fn=mixup_fn,
        model_ema=model_ema,
    )

    print(f"Train Result -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # Assertions to ensure training actually happened and returned valid metrics
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # 5. Evaluation Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 5. Evaluation Demonstration ---")

    # Evaluate using the EMA model
    val_loss, val_acc = evaluate(model_ema.ema_model, val_loader, Config.DEVICE)

    print(f"Val Result -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n--- 6. Inference & Submission ---")

    # Get Test Loader
    test_loader = get_test_loader(img_size=224, batch_size=Config.P1_BATCH_SIZE)

    # Run Inference
    # Returns raw probabilities (N, 5)
    preds = inference(model_ema.ema_model, test_loader, Config.DEVICE, tta=False)

    print(f"Predictions Shape: {preds.shape}")
    assert (
        preds.shape[1] == Config.NUM_CLASSES
    ), "Prediction output has wrong number of classes."

    # Convert probabilities to class labels
    pred_labels = preds.argmax(dim=1).numpy()

    # Create Submission DataFrame
    # We read the test metadata to get image_ids
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Ensure lengths match (test loader might drop last if drop_last=True, but here it is False)
    assert len(df_test) == len(
        pred_labels
    ), "Mismatch between metadata and prediction count."

    submission_df = pd.DataFrame(
        {"image_id": df_test["image_id"], "label": pred_labels}
    )

    # Save Submission
    submission_path = Config.SUBMISSION_FILE
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    # Verify file exists
    assert os.path.exists(submission_path), "Submission file was not created."

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
