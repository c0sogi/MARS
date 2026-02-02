import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    rank_normalize,
    reconstruct_probabilities,
)
from library.dataset import get_dataframes, AppleDataset, get_loaders
from library.model import AppleDiseaseModel
from library.engine import calculate_pos_weights
from library.train import run_training
from library.inference import run_inference


def demo_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n[1] Demonstrating Utils...")

    # 1. Seed Everything
    seed_everything(42)
    print("   Seed set to 42.")

    # 2. Calculate ROC AUC
    y_true = np.array([[0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    auc = calculate_roc_auc(y_true, y_pred)
    print(f"   Calculated ROC AUC: {auc:.4f}")
    assert 0.0 <= auc <= 1.0, "ROC AUC should be between 0 and 1"

    # 3. Rank Normalize
    probs = np.array([[0.1], [0.5], [0.9]])
    ranked = rank_normalize(probs)
    print(f"   Rank Normalized:\n{ranked.flatten()}")
    # Ranks: 0.1->1, 0.5->2, 0.9->3. Normalized (r-1)/(N-1): 0, 0.5, 1.0
    expected_ranks = np.array([[0.0], [0.5], [1.0]])
    np.testing.assert_allclose(ranked, expected_ranks, atol=1e-6)

    # 4. Reconstruct Probabilities
    # Rust=0.8, Scab=0.2
    # Healthy = (1-0.8)*(1-0.2) = 0.2*0.8 = 0.16
    # Multiple = 0.8*0.2 = 0.16
    # RustOnly = 0.8*(1-0.2) = 0.64
    # ScabOnly = (1-0.8)*0.2 = 0.04
    r = np.array([0.8])
    s = np.array([0.2])
    recon = reconstruct_probabilities(r, s)
    print(f"   Reconstructed Probs: {recon[0]}")
    expected_recon = np.array([0.16, 0.16, 0.64, 0.04])
    np.testing.assert_allclose(recon[0], expected_recon, atol=1e-6)
    print("   Utils verification successful.")


def demo_dataset_and_loader():
    """
    Demonstrates dataset loading and dataloader iteration.
    """
    print("\n[2] Demonstrating Dataset & Loader...")

    # Load dataframes (DEBUG mode is on, so this will be fast/small)
    full_train_df, test_df = get_dataframes(load_cached_data=False)
    print(f"   Train DF shape (subset): {full_train_df.shape}")
    print(f"   Test DF shape (subset): {test_df.shape}")

    # Check columns
    required_cols = ["target_rust", "target_scab", "fold", "abs_file_path"]
    for col in required_cols:
        assert col in full_train_df.columns, f"Missing column {col} in train df"

    # Get Loaders for Fold 0
    # Using the overridden model config
    model_config = Config.MODELS[0]
    train_loader, val_loader, test_loader = get_loaders(
        fold=0, model_config=model_config, load_cached_data=True
    )

    # Check batch
    images, targets = next(iter(train_loader))
    print(f"   Batch Image Shape: {images.shape}")
    print(f"   Batch Target Shape: {targets.shape}")

    assert images.shape == (
        model_config["batch_size"],
        3,
        model_config["img_size"],
        model_config["img_size"],
    )
    assert targets.shape == (model_config["batch_size"], 2)  # Rust, Scab
    print("   Dataset and Loader verification successful.")


def demo_model_architecture():
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n[3] Demonstrating Model Architecture...")

    model_name = Config.MODELS[0]["model_name"]
    model = AppleDiseaseModel(model_name=model_name, pretrained=False)
    model.eval()

    # Create dummy input
    img_size = Config.MODELS[0]["img_size"]
    dummy_input = torch.randn(2, 3, img_size, img_size)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Model Output Shape: {output.shape}")
    assert output.shape == (2, 2), "Output shape should be (Batch, Num_Classes)"
    print("   Model architecture verification successful.")


def demo_full_pipeline():
    """
    Runs the full training and inference pipeline using the modified configuration.
    """
    print("\n[4] Running Full Pipeline (Train + Inference)...")

    # 1. Run Training
    # This will train for 1 epoch on fold 0 and save the model
    print("   Starting Training...")
    run_training()

    # Check if model file exists
    expected_model_path = Config.get_model_path(Config.MODELS[0]["model_name"], 0)
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"   Model saved successfully at {expected_model_path}")

    # 2. Run Inference
    # This will load the saved model and generate submission.csv
    print("   Starting Inference...")
    run_inference()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission Shape: {sub_df.shape}")
    print("   Submission Head:")
    print(sub_df.head())

    # Validate submission format
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check if probabilities sum to roughly 1 (optional, but good for sanity)
    # Note: Our reconstruction logic guarantees sum=1 mathematically
    row_sums = sub_df[["healthy", "multiple_diseases", "rust", "scab"]].sum(axis=1)
    np.testing.assert_allclose(
        row_sums.values, 1.0, atol=1e-5, err_msg="Probabilities do not sum to 1"
    )

    print("   Pipeline execution successful.")


if __name__ == "__main__":
    # --- Configuration Overrides for Demo ---
    # We modify the Config class attributes directly to control the execution
    # of the library functions without changing the library code.

    print("Configuring environment for demo...")
    Config.DEBUG = True  # Use small subset of data
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_FOLDS = (
        5  # Keep 5 folds logic, but we will only train specific folds defined in MODELS
    )
    Config.USE_SWA = False  # Disable SWA to save time

    # Set output directory for demo
    Config.IDEA_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "submission.csv")
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Use a lightweight model for demonstration speed (ResNet18)
    # We replace the heavy ensemble with a single fast model
    Config.MODELS = [
        {
            "model_name": "resnet18",
            "img_size": 224,
            "batch_size": 8,
            "fold_indices": [0],  # Only train fold 0
        }
    ]

    # Print config to confirm
    Config.print_config()

    # --- Execute Demos ---
    try:
        demo_utils()
        demo_dataset_and_loader()
        demo_model_architecture()
        demo_full_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\n\n!!! Demo Failed with Error: {e} !!!")
        raise e
