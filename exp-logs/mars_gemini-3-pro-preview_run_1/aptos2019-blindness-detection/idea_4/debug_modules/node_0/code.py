import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import (
    seed_everything,
    quadratic_weighted_kappa,
    decode_ordinal_predictions,
)
from library.dataset import RetinopathyDataset, create_dataloaders
from library.model import OrdinalConvNeXt
from library.trainer import DRTrainer


def run_demo():
    print("=== Starting Demonstration of Diabetic Retinopathy Pipeline ===")

    # 1. Configuration
    SEED = 42
    seed_everything(SEED)

    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Hyperparameters for fast execution
    IMG_SIZE = 224
    BATCH_SIZE = 4
    EPOCHS = 1
    SAMPLE_SIZE = 20  # Small subset for testing

    # Clean working directory if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Verify Utils
    print("\n--- Verifying Utils ---")

    # Test QWK
    y_true = [0, 1, 2, 3, 4]
    y_pred_perfect = [0, 1, 2, 3, 4]
    y_pred_bad = [4, 3, 2, 1, 0]

    score_perfect = quadratic_weighted_kappa(y_true, y_pred_perfect)
    score_bad = quadratic_weighted_kappa(y_true, y_pred_bad)

    print(f"QWK Perfect Score: {score_perfect}")
    assert np.isclose(
        score_perfect, 1.0
    ), "QWK calculation failed for perfect agreement"
    assert score_bad < 1.0, "QWK calculation failed for bad agreement"

    # Test Decoding
    # Create dummy ordinal probs: Batch=2, Classes=4
    # Case 1: High prob for all -> Label 4
    # Case 2: Low prob for all -> Label 0
    dummy_probs = torch.tensor([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    decoded = decode_ordinal_predictions(dummy_probs)
    print(f"Decoded Labels: {decoded}")
    assert decoded[0] == 4, "Decoding logic failed for high probabilities"
    assert decoded[1] == 0, "Decoding logic failed for low probabilities"

    # 3. Verify Dataset and DataLoaders
    print("\n--- Verifying Dataset & DataLoaders ---")

    # Test Dataset instantiation
    ds = RetinopathyDataset(csv_path=TRAIN_CSV, mode="train", sample_size=SAMPLE_SIZE)
    print(f"Dataset length (subset): {len(ds)}")
    assert len(ds) == SAMPLE_SIZE, f"Dataset did not respect sample_size {SAMPLE_SIZE}"

    # Test __getitem__
    img, target = ds[0]
    print(f"Image Shape: {img.shape}, Target: {target}")
    # Image should be (H, W, 3) before transform, but here we didn't pass transform yet so it is numpy array
    # Wait, the dataset class applies transform if provided. If None, it returns raw image.
    # Let's check raw image shape logic in dataset.py: it reads via cv2.
    assert isinstance(
        img, np.ndarray
    ), "Dataset should return numpy array when no transform is passed"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"
    assert target.shape == (4,), "Target ordinal vector should have size 4"

    # Test DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        sample_size=SAMPLE_SIZE,
        seed=SEED,
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        BATCH_SIZE,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), "Incorrect batch image dimensions"
    assert targets.shape == (BATCH_SIZE, 4), "Incorrect batch target dimensions"

    # 4. Verify Model
    print("\n--- Verifying Model ---")
    # Use pretrained=False for speed and to avoid network issues in this demo
    model = OrdinalConvNeXt(model_name="convnext_tiny", pretrained=False, num_classes=4)
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(images)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (BATCH_SIZE, 4), "Model output shape mismatch"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output not in sigmoid range [0, 1]"

    # 5. Verify Trainer (Integration)
    print("\n--- Verifying Trainer (Training & Inference) ---")

    trainer = DRTrainer(
        experiment_dir=WORKING_DIR,
        model_name="convnext_tiny",
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        num_workers=2,  # Reduce workers for demo
        seed=SEED,
    )

    # Override model with non-pretrained for speed in this demo
    trainer.model = OrdinalConvNeXt(
        model_name="convnext_tiny", pretrained=False, num_classes=4
    )
    trainer.model.to(trainer.device)
    # Re-init optimizer since model parameters changed
    import torch.optim as optim

    trainer.optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-4)

    # Run Training
    print("Starting Fit...")
    best_score = trainer.fit(
        train_csv=TRAIN_CSV, val_csv=VAL_CSV, sample_size=SAMPLE_SIZE
    )
    print(f"Fit complete. Best Score: {best_score}")

    assert os.path.exists(
        os.path.join(WORKING_DIR, "best_model.pth")
    ), "best_model.pth not found after training"

    # Run Prediction
    print("Starting Prediction...")
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    trainer.predict_and_submit(test_csv=TEST_CSV, submission_path=submission_path)

    assert os.path.exists(submission_path), "submission.csv not found after prediction"

    # Validate Submission File
    df_sub = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "id_code",
        "diagnosis",
    ], "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"
    # Check if diagnosis values are valid integers 0-4
    assert (
        df_sub["diagnosis"].isin([0, 1, 2, 3, 4]).all()
    ), "Invalid diagnosis values found in submission"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
