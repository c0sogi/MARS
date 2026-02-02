import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import provided library components
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.train import run_training
from library.inference import predict
from library.dataset import RetinopathyDataset, get_transforms
from library.models import RetinopathyModel


def create_balanced_subset(source_csv_path, target_csv_path, samples_per_class=2):
    """
    Creates a small balanced subset of the data to ensure StratifiedKFold
    works correctly even with very few samples.
    """
    if not os.path.exists(source_csv_path):
        raise FileNotFoundError(f"Source file not found: {source_csv_path}")

    df = pd.read_csv(source_csv_path)

    # If diagnosis column exists, sample by class
    if "diagnosis" in df.columns:
        subset_dfs = []
        for label in df["diagnosis"].unique():
            cls_df = df[df["diagnosis"] == label]
            # Sample with replacement if not enough samples
            n = min(len(cls_df), samples_per_class)
            subset_dfs.append(cls_df.sample(n=n, random_state=Config.seed))

        balanced_df = (
            pd.concat(subset_dfs)
            .sample(frac=1, random_state=Config.seed)
            .reset_index(drop=True)
        )
    else:
        # For test set, just take top N
        balanced_df = df.head(samples_per_class * 5)

    balanced_df.to_csv(target_csv_path, index=False)
    print(f"Created subset at {target_csv_path} with {len(balanced_df)} rows.")
    return len(balanced_df)


def main():
    print("=== Starting Diabetic Retinopathy Task Demo ===")

    # 1. Setup and Configuration Overrides
    # We modify the Config singleton directly to control the execution
    seed_everything(42)

    # Define a demo working directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working Directory: {demo_dir}")

    # Override Config parameters for speed and demo purposes
    Config.working_dir = demo_dir
    Config.debug = False  # We will manually create small datasets instead of relying on debug slicing
    Config.image_size = 224  # Smaller image size for faster processing
    Config.batch_size = 4  # Small batch size
    Config.epochs = 1  # Only 1 epoch
    Config.n_folds = 2  # Only 2 folds
    Config.model_archs = ["resnet18"]  # Use a lightweight model
    Config.num_workers = 2

    # 2. Data Preparation (Balanced Subsets)
    print("\n=== Preparing Data Subsets ===")

    # Create temporary CSVs in the working directory
    demo_train_path = os.path.join(demo_dir, "train.csv")
    demo_val_path = os.path.join(demo_dir, "val.csv")
    demo_test_path = os.path.join(demo_dir, "test.csv")

    # We need enough samples to survive a 2-fold split
    # 5 classes * 4 samples = 20 samples total
    n_train = create_balanced_subset(
        Config.train_csv_path, demo_train_path, samples_per_class=4
    )
    n_val = create_balanced_subset(
        Config.val_csv_path, demo_val_path, samples_per_class=2
    )
    n_test = create_balanced_subset(
        Config.test_csv_path, demo_test_path, samples_per_class=2
    )

    # Update Config to point to these new files
    Config.train_csv_path = demo_train_path
    Config.val_csv_path = demo_val_path
    Config.test_csv_path = demo_test_path
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    # 3. Verify Dataset and Transforms Logic
    print("\n=== Verifying Dataset Logic ===")
    df_demo = pd.read_csv(demo_train_path)
    ds = RetinopathyDataset(df_demo, transform=get_transforms("train"), mode="train")
    img, lbl = ds[0]

    # Check Image Tensor
    assert isinstance(img, torch.Tensor), "Dataset should return a torch Tensor"
    assert img.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), f"Expected shape (3, {Config.image_size}, {Config.image_size}), got {img.shape}"
    assert img.dtype == torch.float32, "Image tensor should be float32"

    # Check Label
    assert isinstance(lbl, torch.Tensor), "Label should be a torch Tensor"
    assert lbl.ndim == 0, "Label should be a scalar"
    print("Dataset verification passed.")

    # 4. Verify Model Logic
    print("\n=== Verifying Model Logic ===")
    model = RetinopathyModel(
        "resnet18", pretrained=False
    )  # Pretrained=False to avoid download in strict envs, though True is default
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model architecture verification passed.")

    # 5. Verify Metric Logic
    print("\n=== Verifying Metric Logic ===")
    # Case 1: Perfect agreement
    score_perfect = quadratic_weighted_kappa([0, 1, 2], [0, 1, 2])
    assert np.isclose(
        score_perfect, 1.0
    ), f"Metric should be 1.0 for perfect agreement, got {score_perfect}"

    # Case 2: Random/Bad agreement
    score_bad = quadratic_weighted_kappa([0, 0, 0], [4, 4, 4])
    assert score_bad < 0.5, "Metric should be low for poor agreement"
    print("Metric verification passed.")

    # 6. Run Training Pipeline
    print("\n=== Running Training Pipeline ===")
    # This will train ResNet18 for 1 epoch on 2 folds using the small balanced dataset
    # It saves checkpoints to Config.working_dir
    try:
        run_training("resnet18", epochs=Config.epochs, debug=False)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify checkpoints were created
    for fold in range(Config.n_folds):
        ckpt_path = os.path.join(Config.working_dir, f"resnet18_fold_{fold}.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    print("Training pipeline completed successfully. Checkpoints verified.")

    # 7. Run Inference Pipeline
    print("\n=== Running Inference Pipeline ===")
    # This generates predictions using the checkpoints created above
    # It performs TTA and ensemble averaging
    submission_df = predict(load_cached_data=False, output_path=Config.submission_path)

    # Verify submission file
    assert os.path.exists(Config.submission_path), "Submission file was not created"
    assert (
        len(submission_df) == n_test
    ), f"Submission row count mismatch. Expected {n_test}, got {len(submission_df)}"
    assert "id_code" in submission_df.columns, "Submission missing id_code column"
    assert "diagnosis" in submission_df.columns, "Submission missing diagnosis column"

    # Verify value range
    predictions = submission_df["diagnosis"].values
    assert np.all(predictions >= 0) and np.all(
        predictions <= 4
    ), "Predictions out of range [0, 4]"

    print("Inference pipeline completed successfully.")
    print("\n=== Demo Execution Finished ===")
    print(f"Final submission saved to: {Config.submission_path}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
