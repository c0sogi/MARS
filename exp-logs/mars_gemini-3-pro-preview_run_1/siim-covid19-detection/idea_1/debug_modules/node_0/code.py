import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config, seed_everything
from library.dataset import SIIMDataset, get_transforms
from library.model import MultiTaskUNet
from library.loss import MultiTaskLoss
from library.train import run_training
from library.predict import predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing demonstration script...")

    # 1. Configuration Setup
    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    # Ensure working directory is clean or ready
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print(
        f"Configuration: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, DEVICE={Config.DEVICE}"
    )

    # 2. Dataset & Data Loading Verification
    print("\n--- Verifying Dataset and Transforms ---")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    if Config.DEBUG:
        train_df = train_df.head(Config.MAX_TRAIN_SAMPLES)

    # Instantiate Dataset
    # We force load_cached_data=False initially to test the processing logic
    ds = SIIMDataset(
        df=train_df,
        split="train",
        transform=get_transforms("train"),
        load_cached_data=False,
    )

    print(f"Dataset length: {len(ds)}")
    assert len(ds) > 0, "Dataset should not be empty."

    # Fetch one sample
    sample = ds[0]
    image = sample["image"]
    mask = sample["mask"]
    label = sample["label"]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")
    print(f"Sample Label: {label}")

    # Assertions
    assert image.ndim == 3 and image.shape[0] == 3, "Image should be (3, H, W)"
    assert mask.ndim == 3 and mask.shape[0] == 1, "Mask should be (1, H, W)"
    assert label.shape == (4,), "Label should be (4,)"
    assert image.dtype == torch.float32, "Image should be float32"
    assert mask.dtype == torch.float32, "Mask should be float32"

    print("Dataset verification passed.")

    # 3. Model & Loss Logic Verification
    print("\n--- Verifying Model and Loss Logic ---")
    model = MultiTaskUNet(pretrained=False).to(
        Config.DEVICE
    )  # Pretrained=False for speed in initialization
    criterion = MultiTaskLoss()

    # Prepare batch
    img_batch = image.unsqueeze(0).to(Config.DEVICE)  # (1, 3, H, W)
    mask_batch = mask.unsqueeze(0).to(Config.DEVICE)  # (1, 1, H, W)
    label_batch = label.unsqueeze(0).to(Config.DEVICE)  # (1, 4)

    # Forward pass
    cls_logits, mask_logits = model(img_batch)

    print(f"Logits Shape (Cls): {cls_logits.shape}")
    print(f"Logits Shape (Seg): {mask_logits.shape}")

    # Assertions
    assert cls_logits.shape == (1, 4), "Class logits shape mismatch"
    assert mask_logits.shape == (
        1,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Mask logits shape mismatch"

    # Loss calculation
    loss, metrics = criterion(cls_logits, mask_logits, label_batch, mask_batch)
    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Backward pass check
    model.zero_grad()
    loss.backward()
    print("Backward pass successful.")

    # 4. Training Pipeline Integration
    print("\n--- Running Training Pipeline (Integration Test) ---")
    # We use cached data=True here to speed up if previous step cached it
    run_training(
        debug=True, epochs=1, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Verify checkpoint existence
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."
    print(f"Checkpoint verified at {Config.MODEL_CHECKPOINT_PATH}")

    # 5. Inference Pipeline Integration
    print("\n--- Running Inference Pipeline (Integration Test) ---")
    predict(debug=True, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(sub_df)}")
    print(sub_df.head())

    # Basic format checks
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check if we have both study and image rows
    study_rows = sub_df[sub_df["id"].str.endswith("_study")]
    image_rows = sub_df[sub_df["id"].str.endswith("_image")]

    print(f"Study rows: {len(study_rows)}")
    print(f"Image rows: {len(image_rows)}")

    assert len(study_rows) > 0, "No study predictions found."
    assert len(image_rows) > 0, "No image predictions found."

    print("\nAll integration tests passed successfully.")


if __name__ == "__main__":
    main()
