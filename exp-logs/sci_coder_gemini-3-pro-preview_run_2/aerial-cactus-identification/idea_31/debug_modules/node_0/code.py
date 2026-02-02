import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, load_model
from library.dataset import CactusDataset, get_transforms
from library.model import WideSERes2NeXt
from library.train import run_training
from library.inference import predict


def main():
    print("Initializing Demo Execution...")

    # ==========================================
    # 1. Configuration Override for Speed & Demo
    # ==========================================
    # We modify the Config class directly to limit runtime and isolate outputs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Re-run setup to create the new directories
    Config.setup()

    # Limit training scope
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.MAX_TRAIN_SAMPLES = 100  # Small subset for demonstration
    Config.NUM_WORKERS = 0  # Avoid overhead for small dataset
    Config.SEEDS = [42]  # Single seed for demo

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Max Samples: {Config.MAX_TRAIN_SAMPLES}")

    # ==========================================
    # 2. Dataset & Pipeline Verification
    # ==========================================
    print("\nVerifying Dataset and Data Loading...")

    # Initialize Dataset (Train)
    train_ds = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        phase="train",
        transform=get_transforms("train"),
        max_samples=Config.MAX_TRAIN_SAMPLES,
        load_cached_data=False,  # Force processing to test logic
    )

    # Basic Assertions
    assert (
        len(train_ds) == Config.MAX_TRAIN_SAMPLES
    ), f"Dataset length mismatch. Expected {Config.MAX_TRAIN_SAMPLES}, got {len(train_ds)}"

    # Check Item Structure
    img, label, img_id = train_ds[0]

    # Image should be a Tensor of shape (3, 32, 32)
    assert isinstance(img, torch.Tensor), "Image is not a torch.Tensor"
    assert img.shape == (3, 32, 32), f"Incorrect image shape: {img.shape}"
    assert img.dtype == torch.float32, "Image tensor should be float32"

    # Label should be a float scalar
    assert isinstance(
        label, (float, np.float32)
    ), f"Label is not a float: {type(label)}"

    # ID should be a string
    assert isinstance(img_id, str), "Image ID is not a string"

    print("Dataset verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\nVerifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideSERes2NeXt().to(device)

    # Create dummy input batch (Batch Size=4, Channels=3, H=32, W=32)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check output shape (Batch Size, Num Classes) -> (4, 1)
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"

    print("Model architecture verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\nExecuting Training Loop (1 Epoch)...")

    # Run training for a single seed
    # This uses the parameters we set in Config (EPOCHS=1, etc.)
    seed = 42
    run_training(seed=seed, max_epochs=Config.EPOCHS)

    # Verify model checkpoint was created
    expected_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to generate model checkpoint at {expected_model_path}"
        )

    print(f"Training complete. Model saved to {expected_model_path}")

    # ==========================================
    # 5. Inference & Submission Verification
    # ==========================================
    print("\nVerifying Inference Pipeline...")

    # Load the model we just trained
    model = WideSERes2NeXt().to(device)
    model = load_model(model, seed, device=device)

    # Create Test Loader (Subset)
    test_ds = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        phase="test",
        transform=get_transforms("test"),
        max_samples=50,  # Limit test samples
    )

    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Run Inference
    submission_df = predict(model, test_loader, device)

    # Verify DataFrame Structure
    assert "id" in submission_df.columns, "Submission missing 'id' column"
    assert (
        "has_cactus" in submission_df.columns
    ), "Submission missing 'has_cactus' column"
    assert (
        len(submission_df) == 50
    ), f"Submission length mismatch. Expected 50, got {len(submission_df)}"

    # Verify Probabilities
    probs = submission_df["has_cactus"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    # Save submission to verify file I/O
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Inference verified. Submission saved to {save_path}")

    # ==========================================
    # 6. Metric Utility Verification
    # ==========================================
    print("\nVerifying Metric Calculation...")

    # Synthetic Ground Truth and Predictions
    y_true = np.array([0, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.4])

    # Calculate AUC
    auc = calculate_roc_auc(y_true, y_scores)

    # For this perfect separation, AUC should be 1.0
    assert auc == 1.0, f"AUC calculation incorrect. Expected 1.0, got {auc}"

    # Test with Tensors
    y_true_tensor = torch.tensor(y_true)
    y_scores_tensor = torch.tensor(y_scores)
    auc_tensor = calculate_roc_auc(y_true_tensor, y_scores_tensor)
    assert auc_tensor == 1.0, "AUC calculation with tensors failed"

    print("Metric verification passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)
    main()
