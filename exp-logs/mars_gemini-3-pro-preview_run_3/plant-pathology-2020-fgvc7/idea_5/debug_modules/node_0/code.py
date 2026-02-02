import os
import sys
import pandas as pd
import numpy as np
import torch

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import get_class_weights, calculate_metric
from library.dataset import AppleDataset, get_transforms
from library.models import MultiLevelEfficientNet
from library.trainer import train_fold
from library.inference import generate_submission


def run_pipeline_demonstration():
    print("==== Apple Disease Detection Pipeline Demonstration ====")

    # ------------------------------------------------------------------------
    # 1. Configuration Overrides for Rapid Testing
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for speed...")

    # Set random seed for reproducibility
    seed_everything(42)

    # Override Config parameters to run a minimal version of the pipeline
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Use a very small subset (must be >= batch size)
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.N_FOLDS = 1  # Run only 1 fold instead of 5
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ------------------------------------------------------------------------
    # 2. Verify Utilities and Metadata
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utilities and Metadata...")

    # Load metadata files
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    print(f"    Train Metadata Rows: {len(train_df)}")

    # Test Class Weights Calculation
    weights = get_class_weights(train_df)
    print(f"    Calculated Class Weights: {weights}")

    assert isinstance(weights, torch.Tensor), "Weights must be a torch.Tensor"
    assert (
        weights.shape[0] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} weights"

    # Test Metric Calculation (ROC AUC)
    # Create dummy one-hot targets and probabilities
    y_true_dummy = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_pred_dummy = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )

    metric_score = calculate_metric(y_true_dummy, y_pred_dummy)
    print(f"    Dummy Metric Score: {metric_score:.4f}")
    assert 0.0 <= metric_score <= 1.0, "Metric score out of range"

    # ------------------------------------------------------------------------
    # 3. Verify Dataset and Transforms
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")

    # Create a dataset instance with the debug subset
    debug_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
    dataset = AppleDataset(
        df=debug_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms("train", Config.IMG_SIZE_EFFNET),
        labeled=True,
    )

    # Retrieve one sample
    image, label = dataset[0]

    print(f"    Image Tensor Shape: {image.shape}")
    print(f"    Label Tensor: {label}")

    # Assertions
    assert image.shape == (
        3,
        Config.IMG_SIZE_EFFNET,
        Config.IMG_SIZE_EFFNET,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE_EFFNET}, {Config.IMG_SIZE_EFFNET})"
    assert label.shape == (Config.NUM_CLASSES,), "Label shape mismatch"
    assert isinstance(image, torch.Tensor), "Image is not a tensor"

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture (MultiLevelEfficientNet)...")

    # Instantiate model (pretrained=False for speed in this check)
    model = MultiLevelEfficientNet(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    dummy_batch = torch.randn(2, 3, Config.IMG_SIZE_EFFNET, Config.IMG_SIZE_EFFNET).to(
        Config.DEVICE
    )

    with torch.no_grad():
        output = model(dummy_batch)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # Clean up memory
    del model, dummy_batch, output
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 5. Verify Training Loop (Fold 0)
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Fold 0, 1 Epoch)...")

    # Run training for Fold 0 using EfficientNet
    # This uses the 'debug' flag to slice the dataframe internally in train_fold
    best_auc = train_fold(
        fold_idx=0,
        model_type="effnet",
        train_df=train_df,
        val_df=val_df,
        epochs=Config.EPOCHS,
        debug=True,
    )

    print(f"    Training finished. Best AUC: {best_auc}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "effnet_fold_0_best.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint not created at {checkpoint_path}"

    # ------------------------------------------------------------------------
    # 6. Verify Inference and Submission
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # Generate submission using the trained model
    # Since Config.N_FOLDS is set to 1, it will only look for fold 0 models.
    # It will find 'effnet' fold 0 (created above) and skip 'swin' (not trained).
    generate_submission(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("    Submission File Head:")
    print(submission_df.head())

    # Validate submission format
    assert "image_id" in submission_df.columns, "image_id column missing"
    assert (
        len(submission_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission length {len(submission_df)} does not match debug sample size {Config.DEBUG_SAMPLE_SIZE}"

    # Check probability columns
    for col in Config.CLASSES:
        assert col in submission_df.columns, f"Missing prediction column: {col}"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_pipeline_demonstration()
