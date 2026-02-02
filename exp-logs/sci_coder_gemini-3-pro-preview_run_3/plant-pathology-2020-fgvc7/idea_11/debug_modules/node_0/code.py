import os
import sys
import torch
import pandas as pd
import numpy as np
import timm
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    compute_class_weights,
    ModelEMA,
)
from library.dataset import load_dataset_dfs, AppleDataset, get_transforms
from library.modeling import AppleNet
from library.engine import train_fold, inference


def run_demo():
    print("==== Starting Apple Disease Detection Demo ====")

    # -------------------------------------------------------------------------
    # 1. Override Configuration for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    # Use a lightweight model for the demo to avoid large downloads and slow forward passes
    Config.MODEL_1_NAME = "resnet18"
    Config.MODEL_1_IMG_SIZE = 224
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo
    Config.PATIENCE = 1

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration overridden: ResNet18, 1 Epoch, Batch Size 4.")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test ROC AUC
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    auc_score = calculate_roc_auc(y_true, y_pred)
    assert 0.0 <= auc_score <= 1.0, "ROC AUC score out of range"
    print(f"ROC AUC verification passed. Score: {auc_score:.4f}")

    # Test Class Weights
    # Create a dummy dataframe with known imbalance
    dummy_data = {
        "healthy": [1, 1, 1, 0],
        "multiple_diseases": [0, 0, 0, 0],
        "rust": [0, 0, 0, 1],
        "scab": [0, 0, 0, 0],
    }
    dummy_df = pd.DataFrame(dummy_data)
    # Mock Config.CLASSES for this specific test if needed, but we use the real Config.CLASSES
    # The real classes are ["healthy", "multiple_diseases", "rust", "scab"]
    # Our dummy df matches these columns.

    device = torch.device("cpu")
    weights = compute_class_weights(dummy_df, device, load_cached_data=False)
    assert len(weights) == 4, "Class weights should have 4 elements"
    # Healthy (3) should have lower weight than Rust (1)
    assert (
        weights[0] < weights[2]
    ), "Inverse frequency logic failed: Frequent class should have lower weight"
    print("Class weights verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading and Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[3] Loading and Slicing Data...")
    train_df, val_df, test_df = load_dataset_dfs(load_cached_data=False)

    # Slice dataframes to a tiny subset for the demo
    # Ensure we pick rows where files actually exist (metadata guarantees this, but good to be safe)
    train_subset = train_df.head(20).reset_index(drop=True)
    val_subset = val_df.head(10).reset_index(drop=True)
    test_subset = test_df.head(10).reset_index(drop=True)

    print(
        f"Data sliced: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # Test Dataset Class
    print("Testing AppleDataset...")
    ds = AppleDataset(
        train_subset, transforms=get_transforms("train", Config.MODEL_1_IMG_SIZE)
    )
    sample = ds[0]

    # Check keys
    assert (
        "image" in sample and "target" in sample and "image_id" in sample
    ), "Dataset item missing keys"

    # Check shapes
    img_tensor = sample["image"]
    target_tensor = sample["target"]

    assert img_tensor.shape == (
        3,
        Config.MODEL_1_IMG_SIZE,
        Config.MODEL_1_IMG_SIZE,
    ), f"Incorrect image shape: {img_tensor.shape}"
    assert target_tensor.shape == (4,), f"Incorrect target shape: {target_tensor.shape}"
    assert isinstance(img_tensor, torch.Tensor), "Image is not a tensor"

    print("AppleDataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    model = AppleNet(
        model_name=Config.MODEL_1_NAME, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(2, 3, Config.MODEL_1_IMG_SIZE, Config.MODEL_1_IMG_SIZE)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, 4), got {logits.shape}"
    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch, Subset)...")

    # Define output name for the checkpoint
    demo_model_name = "demo_model_resnet18"

    # Run training
    # This uses the engine.train_fold function which handles the loop, loss, optimizer, etc.
    best_auc = train_fold(
        train_df=train_subset,
        val_df=val_subset,
        model_name=Config.MODEL_1_NAME,
        img_size=Config.MODEL_1_IMG_SIZE,
        output_name=demo_model_name,
    )

    print(f"Training complete. Best AUC on subset: {best_auc:.4f}")

    expected_checkpoint = os.path.join(Config.WORKING_DIR, f"{demo_model_name}.pth")
    assert os.path.exists(expected_checkpoint), "Model checkpoint was not saved."
    print("Checkpoint verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference on Test Subset...")

    predictions = inference(
        model_path=expected_checkpoint,
        test_df=test_subset,
        model_name=Config.MODEL_1_NAME,
        img_size=Config.MODEL_1_IMG_SIZE,
    )

    assert predictions.shape == (
        len(test_subset),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Expected ({len(test_subset)}, 4), got {predictions.shape}"

    # Check if probabilities sum roughly to 1 (Softmax is applied in inference)
    # Note: inference() returns probabilities.
    row_sums = predictions.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Predictions do not sum to 1.0"

    print("Inference verification passed.")
    print("Sample Predictions:\n", predictions[:3])

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
