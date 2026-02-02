import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library import utils, dataset, model, train, predict


def main():
    print("=== Starting Library Demonstration and Verification ===")

    # 1. Configuration Setup for Fast Demonstration
    print("\n[1] Configuring environment for fast execution...")
    # Override Config to use a small subset and minimal training duration
    Config.DEBUG_SAMPLE_SIZE = 50
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = (
        Config.WORKING_DIR
    )  # Save submission in working dir for demo
    Config.SUBMISSION_FILE_PATH = os.path.join(
        Config.SUBMISSION_DIR, "demo_submission.csv"
    )

    # Re-run setup to create the new working directories
    Config.setup()

    # Set seed for reproducibility
    utils.set_seed(Config.SEED)
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying 'utils.py'...")

    # Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Average should be (10*2 + 20*2) / 4 = 60 / 4 = 15
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("AverageMeter logic verified.")

    # Test calculate_accuracy
    # Create dummy logits: Batch size 2, 5 classes
    # Sample 1: Class 2 is highest (Logits: 0, 0, 10, 0, 0) -> Target 2
    # Sample 2: Class 0 is highest (Logits: 10, 0, 0, 0, 0) -> Target 0
    dummy_output = torch.tensor(
        [[0.0, 0.0, 10.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0]]
    )
    dummy_target = torch.tensor([2, 0])

    acc1, acc5 = utils.calculate_accuracy(dummy_output, dummy_target, topk=(1, 5))
    assert acc1 == 100.0, f"calculate_accuracy Top-1 failed. Expected 100.0, got {acc1}"
    assert acc5 == 100.0, f"calculate_accuracy Top-5 failed. Expected 100.0, got {acc5}"
    print("Accuracy calculation verified.")

    # 3. Verify Dataset and DataLoaders
    print("\n[3] Verifying 'dataset.py'...")
    train_loader, val_loader, test_loader = dataset.get_loaders()

    # Verify dataset size matches debug sample size
    assert (
        len(train_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_loader.dataset)}"

    # Fetch a batch to verify shapes
    images, targets = next(iter(train_loader))
    print(f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Check image shape: (B, 3, 224, 224)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.INPUT_SIZE,
        Config.INPUT_SIZE,
    ), "Image tensor shape incorrect."
    # Check target shape: (B,)
    assert targets.shape == (Config.BATCH_SIZE,), "Target tensor shape incorrect."

    # Verify label mapping exists in test loader
    assert hasattr(
        test_loader.dataset, "idx_to_class"
    ), "Test dataset missing 'idx_to_class' attribute."
    print("DataLoaders and Dataset verified.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying 'model.py'...")
    net = model.MobileNetV3Baseline()

    # Move model to CPU for shape verification to avoid unnecessary GPU overhead if not needed yet
    net.eval()

    # Pass the fetched batch through the model
    with torch.no_grad():
        outputs = net(images)

    print(f"Model output shape: {outputs.shape}")
    # Expected output: (Batch Size, Num Classes)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"
    print("Model architecture verified.")

    # 5. Verify Training Pipeline
    print("\n[5] Verifying 'train.py' (Running short training loop)...")

    # We use run_training which encapsulates the whole process.
    # It will use the modified Config values (1 epoch, debug sample size).
    best_acc = train.run_training(
        num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, lr=1e-3, patience=1
    )

    # Check if model checkpoint was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print(f"Training loop completed. Best accuracy: {best_acc}")
    print("Checkpoint creation verified.")

    # 6. Verify Prediction/Inference
    print("\n[6] Verifying 'predict.py'...")

    # Generate predictions using the trained model
    predict.generate_predictions(
        model_path=best_model_path,
        output_path=Config.SUBMISSION_FILE_PATH,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE,
    )

    # Verify submission file exists and has content
    assert os.path.exists(
        Config.SUBMISSION_FILE_PATH
    ), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check columns
    assert (
        "id" in df_sub.columns and "predicted" in df_sub.columns
    ), "Submission file missing required columns."

    # Check if number of rows matches test set size (debug size)
    # Note: test_loader drop_last is False by default, so we get all samples.
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    # Check format of 'predicted' column (should be space-separated integers)
    sample_pred = str(df_sub.iloc[0]["predicted"])
    pred_parts = sample_pred.split(" ")
    assert (
        len(pred_parts) == 5
    ), f"Prediction format incorrect. Expected 5 top-k classes, got {len(pred_parts)} in '{sample_pred}'"

    print("Inference and submission generation verified.")

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")


if __name__ == "__main__":
    main()
