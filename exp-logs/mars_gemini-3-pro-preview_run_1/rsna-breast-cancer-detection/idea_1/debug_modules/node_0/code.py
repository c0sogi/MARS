import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import process_image, probabilistic_f1
from library.dataset import get_dataloaders
from library.model import MetadataEfficientNet
from library.engine import run_training, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Breast Cancer Detection Library Demo...")

    # 1. Setup and Reproducibility
    Config.set_seed(42)

    # Override Config for rapid demonstration
    Config.BATCH_SIZE = 4
    DEMO_SAMPLE_SIZE = 20
    DEMO_EPOCHS = 1

    print(f"\n[1] Configuration Overrides:")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Sample Size: {DEMO_SAMPLE_SIZE}")
    print(f"    Epochs: {DEMO_EPOCHS}")

    # 2. Verify Utils: Metric Calculation
    print("\n[2] Verifying Utils (Probabilistic F1)...")
    # Case 1: Perfect prediction
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred_perfect = torch.tensor([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected pF1=1.0 for perfect pred, got {score_perfect}"

    # Case 2: Complete failure
    y_pred_wrong = torch.tensor([0.0, 1.0, 0.0, 1.0])
    score_wrong = probabilistic_f1(y_true, y_pred_wrong)
    assert np.isclose(
        score_wrong, 0.0
    ), f"Expected pF1=0.0 for wrong pred, got {score_wrong}"

    print("    pF1 Metric logic verified.")

    # 3. Verify Utils: Image Processing
    print("\n[3] Verifying Image Processing...")
    # Get a valid file path from metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if not df_train.empty:
        sample_path = df_train.iloc[0]["file_path"]
        print(f"    Processing sample image: {sample_path}")

        # Process image (Load -> ROI Crop -> Resize -> Normalize)
        processed_img = process_image(sample_path)

        # Check shape (H, W) - process_image returns 2D array
        expected_shape = Config.IMG_SIZE
        assert (
            processed_img.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {processed_img.shape}"

        # Check normalization [0, 1]
        assert (
            0.0 <= processed_img.min() and processed_img.max() <= 1.0
        ), "Image values not normalized to [0, 1]"

        print(
            f"    Image processed successfully. Shape: {processed_img.shape}, Range: [{processed_img.min():.2f}, {processed_img.max():.2f}]"
        )
    else:
        print("    Warning: Train metadata is empty, skipping image check.")

    # 4. Verify Dataset and DataLoaders
    print("\n[4] Verifying Dataset and DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, sample_size=DEMO_SAMPLE_SIZE
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    # Verify Input Shape: (Batch, Channels, H, W)
    # Channels should be 3 (1 Image + 1 Age + 1 Implant)
    expected_channels = Config.TOTAL_INPUT_CHANNELS
    assert inputs.shape == (
        Config.BATCH_SIZE,
        expected_channels,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, expected_channels, Config.IMG_SIZE[0], Config.IMG_SIZE[1])}, got {inputs.shape}"

    # Verify Targets
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    # Verify Metadata Channel Content
    # Channel 1 is Age (normalized), Channel 2 is Implant (0 or 1)
    age_channel = inputs[0, 1, :, :]
    implant_channel = inputs[0, 2, :, :]

    # Age channel should be constant across spatial dims
    assert torch.std(age_channel) < 1e-6, "Age channel should be spatially constant"
    assert (
        torch.std(implant_channel) < 1e-6
    ), "Implant channel should be spatially constant"

    print(f"    Batch loaded successfully. Input Shape: {inputs.shape}")

    # 5. Verify Model Architecture
    print("\n[5] Verifying Model Architecture...")
    model = MetadataEfficientNet(
        pretrained=False
    )  # No need to download weights for shape check
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    # Output shape should be (Batch, 1)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {outputs.shape}"

    print("    Model forward pass successful.")

    # 6. Verify Training Loop (Engine)
    print("\n[6] Running Training Loop (Demo)...")
    # Note: run_training uses library.model.run_training which handles device placement
    trained_model = run_training(sample_size=DEMO_SAMPLE_SIZE, epochs=DEMO_EPOCHS)

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print(f"    Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # 7. Verify Inference and Submission
    print("\n[7] Generating Submission...")
    generate_submission(model=trained_model, sample_size=DEMO_SAMPLE_SIZE)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(df_sub)} rows.")

    # Check submission format
    assert (
        "prediction_id" in df_sub.columns and "cancer" in df_sub.columns
    ), "Submission file missing required columns."

    # Check probability range
    preds = df_sub["cancer"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
