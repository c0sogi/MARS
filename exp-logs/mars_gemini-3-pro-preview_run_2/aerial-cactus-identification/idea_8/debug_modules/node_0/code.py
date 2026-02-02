import os
import shutil
import torch
import pandas as pd
import numpy as np

# Importing provided library components
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import LightweightPyramidNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # Configuration
    DEMO_SEED = 42
    WORK_DIR = "./working/demo_run"

    # 1. Setup Environment
    print(">>> Setting up environment...")
    set_seed(DEMO_SEED)
    device = get_device()
    print(f"Device: {device}")

    # Clean/Create working directory for this demo run
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    # 2. Dataset Verification
    print("\n>>> Verifying Dataset Loading...")
    # Using a small batch size for quick verification
    # load_cached_data=False ensures we test the raw data loading pipeline at least once
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=16, num_workers=2, load_cached_data=False, seed=DEMO_SEED
    )

    # Check Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == (16, 3, 32, 32), "Train image batch shape mismatch"
    assert labels.shape == (16,), "Train label batch shape mismatch"
    assert images.dtype == torch.float32, "Images should be float32"
    # Check normalization roughly (standard ToTensor puts it in [0, 1])
    assert images.max() <= 1.0 and images.min() >= 0.0, "Images not normalized to [0,1]"

    # Check Test Loader
    test_images, _ = next(iter(test_loader))
    print(f"Test Batch - Images: {test_images.shape}")
    assert test_images.shape == (16, 3, 32, 32), "Test image batch shape mismatch"
    assert len(test_ids) == 3325, f"Expected 3325 test IDs, got {len(test_ids)}"

    print("Dataset verification passed.")

    # 3. Model Verification
    print("\n>>> Verifying Model Architecture...")
    model = LightweightPyramidNet().to(device)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Ensure model is in eval mode for deterministic check
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Model output shape mismatch, expected (4, 1)"
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print("Model verification passed.")

    # 4. Training Pipeline Verification
    print("\n>>> Verifying Training Pipeline...")
    # Run a minimal training session: 2 epochs to verify loop and scheduler stepping
    # We use a larger batch size (128) to speed up the epoch
    model_path, best_auc = run_training(
        seed=DEMO_SEED,
        epochs=2,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-2,
        patience=2,
        save_dir=WORK_DIR,
    )

    print(f"Training finished. Model saved to: {model_path}")
    print(f"Best Validation AUC: {best_auc}")

    assert os.path.exists(model_path), "Model file was not created"
    assert isinstance(best_auc, float), "AUC is not a float"
    assert 0.0 <= best_auc <= 1.0, "AUC is out of valid range [0, 1]"
    print("Training pipeline verification passed.")

    # 5. Inference Pipeline Verification
    print("\n>>> Verifying Inference Pipeline...")
    submission_path = os.path.join(WORK_DIR, "submission", "submission.csv")

    generate_submission(
        model_paths=[model_path],
        output_file=submission_path,
        batch_size=128,
        num_workers=2,
        seed=DEMO_SEED,
    )

    assert os.path.exists(submission_path), "Submission file was not generated"

    # Validate Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")

    assert df_sub.shape == (
        3325,
        2,
    ), f"Submission shape mismatch. Expected (3325, 2), got {df_sub.shape}"
    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns mismatch"
    assert not df_sub.isnull().values.any(), "Submission contains null values"

    # Check probability range
    probs = df_sub["has_cactus"].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Verify IDs match
    assert set(df_sub["id"].values) == set(
        test_ids
    ), "Submission IDs do not match Test IDs"

    print("Inference pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
