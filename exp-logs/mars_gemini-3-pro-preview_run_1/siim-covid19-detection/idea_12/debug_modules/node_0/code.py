import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import ResNet18D_UNet
from library.engine import train_one_epoch, validate
from library.inference import predict_and_submit


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup and Configuration Override
    # We override Config parameters to run a fast demo in the working directory
    print("\n[1] Configuring environment for demo...")

    Config.debug = True  # Use subset of data (100 samples)
    Config.epochs = 1  # Run only 1 epoch
    Config.train_batch_size = 8
    Config.valid_batch_size = 8

    # Define working directories for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.cache_dir = demo_dir
    Config.submission_dir = demo_dir
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    seed_everything(Config.seed)
    print(f"Debug Mode: {Config.debug}")
    print(f"Cache Dir: {Config.cache_dir}")

    # 2. Data Loading
    print("\n[2] Loading Data...")
    # This will trigger processing and caching of the debug subset
    train_loader, val_loader, test_loader = get_loaders(
        debug=Config.debug, load_cached_data=False
    )

    # Verify Train Loader
    print("Verifying Train Loader batch structure...")
    images, labels, masks = next(iter(train_loader))

    # Assertions for shapes
    # Images: (B, 3, H, W) -> (8, 3, 512, 512)
    assert (
        images.dim() == 4 and images.shape[1] == 3
    ), f"Image shape mismatch: {images.shape}"
    assert images.shape[2] == Config.img_size and images.shape[3] == Config.img_size

    # Labels: (B, Num_Classes) -> (8, 4)
    assert (
        labels.dim() == 2 and labels.shape[1] == Config.num_study_classes
    ), f"Label shape mismatch: {labels.shape}"

    # Masks: (B, 1, H, W) -> (8, 1, 512, 512)
    assert (
        masks.dim() == 4 and masks.shape[1] == 1
    ), f"Mask shape mismatch: {masks.shape}"

    print(
        f"Batch shapes confirmed: Images {images.shape}, Labels {labels.shape}, Masks {masks.shape}"
    )

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    # We use pretrained=False to avoid download overhead/errors in this demo environment
    model = ResNet18D_UNet(num_classes=Config.num_study_classes, pretrained=False)
    model.to(Config.device)

    # Verify Model Forward Pass
    print("Verifying Model Forward Pass...")
    dummy_input = images.to(Config.device)
    with torch.no_grad():
        logits, pred_masks = model(dummy_input)

    # Check outputs
    assert logits.shape == (
        images.shape[0],
        Config.num_study_classes,
    ), f"Logits shape incorrect: {logits.shape}"
    assert pred_masks.shape == (
        images.shape[0],
        1,
        Config.img_size,
        Config.img_size,
    ), f"Pred masks shape incorrect: {pred_masks.shape}"

    print("Model forward pass successful.")

    # 4. Training Loop Demo
    print("\n[4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # Train for one epoch
    train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        device=Config.device,
        epoch=1,
    )

    assert not np.isnan(train_loss), "Training loss returned NaN"
    print(f"Training completed. Loss: {train_loss:.4f}")

    # 5. Validation Demo
    print("\n[5] Running Validation...")
    val_loss, val_map = validate(model, val_loader, Config.device)

    print(f"Validation completed. Loss: {val_loss:.4f}, mAP: {val_map:.4f}")
    assert val_loss >= 0, "Validation loss should be non-negative"
    assert 0.0 <= val_map <= 1.0, "mAP score should be between 0 and 1"

    # 6. Inference and Submission
    print("\n[6] Running Inference and Generating Submission...")

    # Save the current model state to simulate loading a trained model
    model_path = os.path.join(Config.cache_dir, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Run the inference pipeline
    # This function handles TTA, post-processing, and CSV generation
    predict_and_submit(model_path=model_path)

    # 7. Verify Submission File
    print("\n[7] Verifying Submission File...")
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    df_sub = pd.read_csv(Config.submission_path)
    print(f"Submission loaded. Rows: {len(df_sub)}")
    print("Head:")
    print(df_sub.head())

    # Check columns
    expected_cols = ["Id", "PredictionString"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check if we have predictions for the test set (debug size)
    # Note: Test set in debug mode is 100 images.
    # Submission has rows for studies and images.
    # We just ensure it's not empty.
    assert len(df_sub) > 0, "Submission file is empty"

    # Check format of PredictionString
    sample_pred = df_sub.iloc[0]["PredictionString"]
    assert isinstance(sample_pred, str), "PredictionString is not a string"
    assert (
        len(sample_pred.split()) >= 6
    ), "PredictionString format seems too short (expected class conf x y w h ...)"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
