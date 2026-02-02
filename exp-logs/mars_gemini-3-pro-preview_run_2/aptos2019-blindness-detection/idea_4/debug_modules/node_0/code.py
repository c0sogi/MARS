import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa, MetricMonitor
from library.dataset import RetinopathyDataset, get_transforms
from library.models import RetinopathyModel
from library.train import train_models
from library.inference import predict_and_submit


def run_demo():
    print("=== Starting Demonstration of Diabetic Retinopathy Pipeline ===")

    # 1. Setup and Configuration Overrides for Demo
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(42)

    # Override Config for speed and isolation
    # We use a lightweight model (ResNet18) and minimal training steps
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.NUM_FOLDS = 2  # Reduce folds to 2 for speed
    Config.BATCH_SIZE = 8
    Config.MODEL_SPECS = {"resnet18": 224}  # Use a lightweight model supported by timm

    # Use a specific working directory for this demo
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"

    # Ensure directories exist and are clean
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Output Directory: {Config.OUTPUT_DIR}")
    print(f"Model Specs: {Config.MODEL_SPECS}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test Quadratic Weighted Kappa
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred_perfect = np.array([0, 1, 2, 3, 4])
    y_pred_bad = np.array([4, 3, 2, 1, 0])

    kappa_perfect = quadratic_weighted_kappa(y_true, y_pred_perfect)
    kappa_bad = quadratic_weighted_kappa(y_true, y_pred_bad)

    assert np.isclose(kappa_perfect, 1.0), f"Expected Kappa 1.0, got {kappa_perfect}"
    assert (
        kappa_bad < 0.0
    ), f"Expected negative Kappa for inverse predictions, got {kappa_bad}"
    print("  > Quadratic Weighted Kappa logic verified.")

    # Test MetricMonitor
    monitor = MetricMonitor()
    monitor.update("loss", 10.0)
    monitor.update("loss", 5.0)
    assert monitor.get_avg("loss") == 7.5, "MetricMonitor average calculation failed."
    print("  > MetricMonitor logic verified.")

    # 3. Verify Dataset and Transforms
    print("\n[3] Verifying Dataset and Transforms...")

    # Create a dummy dataset instance using the provided metadata
    # We use the training metadata provided in the environment
    ds = RetinopathyDataset(
        csv_path=Config.TRAIN_CSV,
        transform=get_transforms(image_size=224, mode="train"),
        mode="train",
    )

    # Assert dataset is not empty
    assert len(ds) > 0, "Dataset is empty."

    # Fetch one sample
    image, label = ds[0]

    # Check types and shapes
    assert isinstance(image, torch.Tensor), "Image is not a tensor."
    # Shape should be (Channels, Height, Width)
    assert image.shape == (3, 224, 224), f"Unexpected image shape: {image.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor."
    # Label is a scalar float tensor for regression
    assert label.ndim == 0, "Label should be a scalar."

    print(
        f"  > Dataset sample check passed. Image shape: {image.shape}, Label: {label.item()}"
    )

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")

    # Instantiate the lightweight model defined in Config override
    # pretrained=False to avoid downloading weights during this unit test (train loop uses True)
    model = RetinopathyModel(model_name="resnet18", pretrained=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size,) because the model flattens the output
    assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"
    print("  > Model forward pass verified.")

    # 5. Run Training Pipeline (Integration Test)
    print("\n[5] Running Training Pipeline (Debug Mode)...")

    # This will train ResNet18 for 1 epoch on 2 folds using a subset of data (100 samples)
    # It saves models to Config.OUTPUT_DIR
    train_models(debug=True)

    # Verify artifacts
    expected_models = [f"resnet18_fold_0.pth", f"resnet18_fold_1.pth"]

    for model_file in expected_models:
        path = os.path.join(Config.OUTPUT_DIR, model_file)
        assert os.path.exists(path), f"Training failed: Model file {path} not found."

    print("  > Training pipeline completed successfully. Checkpoints generated.")

    # 6. Run Inference Pipeline (Integration Test)
    print("\n[6] Running Inference Pipeline...")

    # This will load the trained models and generate a submission
    # It uses Config.MODEL_SPECS to know which models to load
    predict_and_submit(debug=True)

    # Verify submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Inference failed: {submission_path} not found."

    # Check submission content
    df_sub = pd.read_csv(submission_path)
    assert (
        "id_code" in df_sub.columns and "diagnosis" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check diagnosis values are integers between 0 and 4
    assert pd.api.types.is_integer_dtype(
        df_sub["diagnosis"]
    ), "Diagnosis must be integer."
    assert (
        df_sub["diagnosis"].min() >= 0 and df_sub["diagnosis"].max() <= 4
    ), "Diagnosis values out of range."

    print("  > Inference pipeline completed successfully.")
    print(f"  > Submission saved to {submission_path}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
