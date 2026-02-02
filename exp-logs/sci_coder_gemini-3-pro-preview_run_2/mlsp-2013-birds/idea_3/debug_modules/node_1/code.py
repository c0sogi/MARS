import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import BirdDataset, get_transforms, load_data
from library.model import BirdClassifier
from library.train import run_kfold_training
from library.inference import predict_and_submit


def main():
    print("Starting Demonstration of Bird Species Classification Pipeline...")

    # 1. Reproducibility
    seed_everything(Config.SEED)
    print("\n[1] Seed set for reproducibility.")

    # 2. Verify Dataset and Transforms
    print("\n[2] Verifying Dataset and Transforms...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_CSV_PATH}")

    df_train = pd.read_csv(Config.TRAIN_CSV_PATH)

    # Use a small subset for verification
    df_subset = df_train.head(10).reset_index(drop=True)

    # Instantiate Dataset
    dataset = BirdDataset(df_subset, transforms=get_transforms(data="train"))

    # Fetch one sample
    image, label = dataset[0]

    # Verify Shapes
    # Image: (Channels, Height, Width) -> (3, 224, 224) defined in Config
    expected_img_shape = (Config.CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    expected_label_shape = (Config.NUM_CLASSES,)

    print(f"   Image Shape: {image.shape}")
    print(f"   Label Shape: {label.shape}")

    assert (
        image.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {image.shape}"
    assert (
        label.shape == expected_label_shape
    ), f"Label shape mismatch. Expected {expected_label_shape}, got {label.shape}"
    assert isinstance(image, torch.Tensor), "Image is not a Tensor"
    assert isinstance(label, torch.Tensor), "Label is not a Tensor"

    print("   Dataset verification passed.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")

    model = BirdClassifier(
        backbone=Config.BACKBONE,
        pretrained=False,  # Speed up initialization, no download needed for shape check
        num_classes=Config.NUM_CLASSES,
    )
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(
        batch_size, Config.CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
    ).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"   Input Batch Shape: {dummy_input.shape}")
    print(f"   Output Logits Shape: {outputs.shape}")

    assert outputs.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {outputs.shape}"

    print("   Model architecture verification passed.")

    # 4. Run Training Pipeline (Mini-Scale)
    print("\n[4] Running Training Pipeline (Debug Mode)...")

    # We use run_kfold_training with debug=True to limit data size
    # We set epochs=1 and n_folds=2 to ensure it runs quickly but tests the fold logic
    try:
        run_kfold_training(debug=True, epochs=1, batch_size=4, n_folds=2)
        print("   Training pipeline executed successfully.")
    except Exception as e:
        print(f"   Training pipeline failed: {e}")
        raise e

    # 5. Verify Inference and Submission
    print("\n[5] Verifying Inference and Submission Generation...")

    # Run the inference module explicitly
    # This uses the models saved in step 4
    try:
        predict_and_submit(n_folds=2, debug=True)
        print("   Inference pipeline executed successfully.")
    except Exception as e:
        print(f"   Inference pipeline failed: {e}")
        raise e

    # Check if submission file exists
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"   Submission file found at: {submission_path}")

        # Verify content format
        df_sub = pd.read_csv(submission_path)
        print(f"   Submission Rows: {len(df_sub)}")
        print(f"   Submission Columns: {list(df_sub.columns)}")

        required_cols = ["Id", "Probability"]
        assert all(
            col in df_sub.columns for col in required_cols
        ), "Missing required columns in submission file."
        assert not df_sub.isnull().values.any(), "Submission file contains NaNs."

        # Check Id format (should be integer)
        assert pd.api.types.is_integer_dtype(
            df_sub["Id"]
        ), "Id column should be integer."

        # Check Probability range
        probs = df_sub["Probability"]
        assert (probs >= 0).all() and (
            probs <= 1
        ).all(), "Probabilities out of range [0, 1]."

        print("   Submission content verification passed.")
    else:
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    print("\n" + "=" * 50)
    print("All demonstrations and verifications completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    main()
