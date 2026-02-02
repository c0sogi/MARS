import os
import torch
import pandas as pd
import numpy as np
from library import utils, dataset, model, trainer, inference


def main():
    print("=== Iceberg Classifier Library Demo ===")

    # 1. Setup
    print("\n[1] Initializing Environment...")
    utils.seed_everything(42)
    device = utils.get_device()
    print(f"Device: {device}")

    # 2. Data Pipeline
    print("\n[2] Verifying Data Pipeline...")
    # We set load_cached_data=False to ensure we test the raw data processing logic
    # and generate the specific cache files expected by the library (X_train.npy, etc.)
    # The batch size is small for demonstration purposes.
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=16,
        num_workers=0,  # Use 0 workers to avoid overhead in this short script
        load_cached_data=False,
    )

    # Fetch a batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]

    print(
        f"Train Batch - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert images.shape == (
        16,
        3,
        75,
        75,
    ), f"Expected (16, 3, 75, 75), got {images.shape}"
    assert angles.shape == (16,), f"Expected (16,), got {angles.shape}"
    assert labels.shape == (16,), f"Expected (16,), got {labels.shape}"
    print("Data loading verified.")

    # 3. Model Architecture
    print("\n[3] Verifying Model Architecture...")
    net = model.MicroResNet().to(device)

    # Dummy forward pass
    img_gpu = images.to(device)
    ang_gpu = angles.to(device)

    with torch.no_grad():
        output = net(img_gpu, ang_gpu)

    print(f"Model Output Shape: {output.shape}")
    print(f"Sample Predictions: {output[:5].cpu().numpy()}")

    # Assertions
    assert output.shape == (16,), f"Expected output shape (16,), got {output.shape}"
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Predictions must be probabilities [0, 1]"
    print("Model architecture verified.")

    # 4. Training Loop (Cross-Validation)
    print("\n[4] Running Training Demonstration (2 Folds, 1 Epoch)...")
    # We use n_splits=2 and epochs=1 to keep execution time minimal (~seconds)
    # load_cached_data=True utilizes the cache generated in Step 2
    angle_mean = trainer.run_cross_validation(
        n_splits=2,
        epochs=1,
        batch_size=32,
        lr=1e-3,
        patience=1,
        num_workers=0,
        load_cached_data=True,
        seed=42,
    )

    # Verify checkpoints were created
    fold_0_path = "./working/idea_5/fold_0/model_best.pth"
    fold_1_path = "./working/idea_5/fold_1/model_best.pth"

    assert os.path.exists(fold_0_path), f"Checkpoint missing: {fold_0_path}"
    assert os.path.exists(fold_1_path), f"Checkpoint missing: {fold_1_path}"
    print(f"Training complete. Checkpoints verified. Angle mean: {angle_mean:.4f}")

    # 5. Inference
    print("\n[5] Running Inference Demonstration...")
    submission_path = "./working/demo_submission.csv"

    inference.generate_submission(
        n_splits=2,
        batch_size=32,
        num_workers=0,
        load_cached_data=True,
        angle_mean=angle_mean,
        output_path=submission_path,
    )

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Assertions on submission content
    assert len(df_sub) == 321, f"Expected 321 rows (test set size), got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect columns in submission"
    assert df_sub["is_iceberg"].between(0, 1).all(), "Probabilities out of bounds"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
