import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # --- 1. Setup and Configuration Override ---
    print("\n[1] Configuring environment for demo...")

    # Set a fixed seed for reproducibility
    seed_everything(42)

    # Override Config for speed and demo purposes
    # We modify the class attributes directly.
    Config.debug = True  # Use a small subset of data (500 train, 100 val)
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 2  # Reduce workers for simple demo
    Config.exp_name = "demo_run"  # Separate experiment name
    Config.working_dir = f"./working/{Config.exp_name}"

    # Ensure working directory is clean
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Debug Mode: {Config.debug}")
    print(f"Epochs: {Config.epochs}")
    print(f"Working Directory: {Config.working_dir}")

    # --- 2. Dataset and DataLoader Verification ---
    print("\n[2] Verifying Dataset and DataLoader...")

    # Load metadata (using the paths defined in Config)
    df_train = pd.read_csv(Config.train_metadata_path)

    # Instantiate dataset with training transforms
    dataset = CatheterDataset(df_train.head(10), transform=get_transforms("train"))

    # Fetch a single item
    image, label = dataset[0]

    # Verification
    print(f"Image Shape: {image.shape}")
    print(f"Label Shape: {label.shape}")

    # Assertions
    assert isinstance(image, torch.Tensor), "Image should be a torch Tensor"
    assert image.shape == (
        3,
        640,
        640,
    ), f"Expected image shape (3, 640, 640), got {image.shape}"
    assert label.shape == (11,), f"Expected label shape (11,), got {label.shape}"
    assert image.dtype == torch.float32, "Image dtype should be float32"

    print("Dataset verification passed.")

    # --- 3. Model Architecture Verification ---
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate model
    model = CatheterModel(
        model_name=Config.model_name,
        pretrained=False,  # Disable pretrained weights download for speed/offline check
        num_classes=Config.num_classes,
    )
    model.to(device)
    model.eval()

    # Create a dummy batch
    dummy_input = torch.randn(2, 3, 640, 640).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, 11), f"Expected output shape (2, 11), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model verification passed.")

    # --- 4. Metric Logic Verification ---
    print("\n[4] Verifying Metric Calculation (AUC)...")

    # Scenario 1: Perfect prediction
    y_true_perfect = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y_pred_perfect = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    score_perfect = get_score(y_true_perfect, y_pred_perfect)
    print(f"Perfect Score (Expected ~1.0): {score_perfect}")
    assert score_perfect == 1.0, "Metric calculation failed for perfect predictions"

    # Scenario 2: Constant label column (Edge case handling)
    # Column 0 has both 0 and 1. Column 1 has only 0s.
    y_true_edge = np.array([[0, 0], [1, 0], [0, 0], [1, 0]])
    y_pred_edge = np.array([[0.2, 0.2], [0.8, 0.2], [0.3, 0.2], [0.7, 0.2]])

    # The function should return 0.5 for the constant column and the actual AUC for the valid column
    # AUC for col 0 (0,1,0,1 vs 0.2,0.8,0.3,0.7) -> 1.0
    # AUC for col 1 (constant) -> 0.5
    # Average -> 0.75
    score_edge = get_score(y_true_edge, y_pred_edge)
    print(f"Edge Case Score (Expected 0.75): {score_edge}")
    assert (
        score_edge == 0.75
    ), f"Metric calculation failed for constant label column. Got {score_edge}"

    print("Metric verification passed.")

    # --- 5. Full Training Pipeline Execution ---
    print("\n[5] Executing Full Training Pipeline (Debug Mode)...")

    # This function handles data loading, model init, training loop, validation,
    # saving best model, and generating submission.
    best_auc = run_training()

    print(f"Training pipeline finished. Best Validation AUC: {best_auc}")

    # --- 6. Output Artifact Verification ---
    print("\n[6] Verifying Output Artifacts...")

    expected_model_path = os.path.join(Config.working_dir, "best_model.pth")
    expected_submission_path = "./submission/submission.csv"

    # Verify Model File
    if os.path.exists(expected_model_path):
        print(f"SUCCESS: Model file found at {expected_model_path}")
        file_size = os.path.getsize(expected_model_path)
        print(f"Model file size: {file_size / (1024*1024):.2f} MB")
        assert file_size > 0, "Model file is empty"
    else:
        raise FileNotFoundError(f"Model file not found at {expected_model_path}")

    # Verify Submission File
    if os.path.exists(expected_submission_path):
        print(f"SUCCESS: Submission file found at {expected_submission_path}")
        df_sub = pd.read_csv(expected_submission_path)
        print(f"Submission shape: {df_sub.shape}")

        # Check columns
        expected_cols = ["StudyInstanceUID"] + Config.target_cols
        assert (
            list(df_sub.columns) == expected_cols
        ), "Submission columns do not match requirements"

        # Check row count (Should match test metadata)
        df_test_meta = pd.read_csv(Config.test_metadata_path)
        assert len(df_sub) == len(
            df_test_meta
        ), f"Submission row count ({len(df_sub)}) does not match test set size ({len(df_test_meta)})"

        # Check values are probabilities
        numeric_cols = df_sub[Config.target_cols]
        assert (numeric_cols >= 0).all().all() and (
            numeric_cols <= 1
        ).all().all(), "Submission contains values outside [0, 1] probability range"

    else:
        raise FileNotFoundError(
            f"Submission file not found at {expected_submission_path}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
