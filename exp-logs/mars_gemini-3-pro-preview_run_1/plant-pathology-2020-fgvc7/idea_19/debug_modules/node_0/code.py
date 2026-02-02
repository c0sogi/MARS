import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
from sklearn.model_selection import train_test_split

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import WeightedSoftTargetCrossEntropy
import library.trainer as trainer
import library.inference as inference


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup Configuration for Demo (Optimize for Speed)
    # We modify the Config singleton directly to affect all imported modules
    print("\n[1] Configuring environment for rapid demonstration...")

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters
    Config.working_dir = DEMO_DIR
    Config.checkpoint_dir = os.path.join(DEMO_DIR, "models")
    Config.output_dir = os.path.join(DEMO_DIR, "output")
    Config.submission_dir = os.path.join(DEMO_DIR, "submission")
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Create necessary directories
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)
    os.makedirs(os.path.join(DEMO_DIR, "cache"), exist_ok=True)

    # Reduce computational load
    Config.epochs = 2
    Config.n_folds = 2
    Config.batch_size = 4
    Config.img_size = 128  # Smaller image size for speed
    Config.num_workers = 0  # Avoid multiprocessing overhead for tiny datasets
    Config.debug = True

    seed_everything(Config.seed)
    print("Configuration updated for demo mode.")

    # 2. Prepare Data Subsets
    print("\n[2] Preparing data subsets...")

    # Load original metadata
    df_train_full = pd.read_csv("./metadata/train_metadata.csv")
    df_test_full = pd.read_csv("./metadata/test_metadata.csv")

    # Create a stratified subset for training (ensuring all classes are present)
    # We take 8 samples per class to ensure we have enough for 2 folds
    subset_indices = []
    for label in df_train_full["stratify_label"].unique():
        indices = df_train_full[df_train_full["stratify_label"] == label].index
        if len(indices) >= 8:
            subset_indices.extend(indices[:8])
        else:
            subset_indices.extend(indices)

    df_train_subset = df_train_full.loc[subset_indices].reset_index(drop=True)
    df_test_subset = df_test_full.head(10).reset_index(
        drop=True
    )  # First 10 test images

    # Save subsets to demo directory
    train_subset_path = os.path.join(DEMO_DIR, "train_subset.csv")
    val_subset_path = os.path.join(
        DEMO_DIR, "val_subset.csv"
    )  # We'll just duplicate for simplicity of the 'get_all_data' logic
    test_subset_path = os.path.join(DEMO_DIR, "test_subset.csv")

    # In the trainer.py logic, it merges train and val metadata.
    # We will split our subset into train/val files to mimic the directory structure expected.
    df_tr, df_val = train_test_split(
        df_train_subset,
        test_size=0.5,
        stratify=df_train_subset["stratify_label"],
        random_state=42,
    )

    df_tr.to_csv(train_subset_path, index=False)
    df_val.to_csv(val_subset_path, index=False)
    df_test_subset.to_csv(test_subset_path, index=False)

    # Point Config to these new files
    Config.train_metadata_path = train_subset_path
    Config.val_metadata_path = val_subset_path
    Config.test_metadata_path = test_subset_path

    print(f"Created train subset: {len(df_tr)} samples")
    print(f"Created val subset: {len(df_val)} samples")
    print(f"Created test subset: {len(df_test_subset)} samples")

    # 3. Verify Dataset Class
    print("\n[3] Verifying AppleDataset...")
    dataset = AppleDataset(df_tr, transform=get_transforms("train"))
    image, target = dataset[0]

    # Assertions
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert (
        image.ndim == 3
    ), f"Image should have 3 dimensions (C, H, W), got {image.ndim}"
    assert image.shape[0] == 3, f"Image should have 3 channels, got {image.shape[0]}"
    assert (
        image.shape[1] == Config.img_size and image.shape[2] == Config.img_size
    ), f"Image size mismatch. Expected {Config.img_size}x{Config.img_size}, got {image.shape[1]}x{image.shape[2]}"
    assert isinstance(target, torch.Tensor), "Target should be a torch.Tensor"
    assert (
        target.shape[0] == Config.num_classes
    ), f"Target shape mismatch. Expected {Config.num_classes}, got {target.shape[0]}"

    print("AppleDataset verification passed.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")
    model = get_model(
        Config.model_name, pretrained=False, num_classes=Config.num_classes
    )
    model.to(Config.device)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.img_size, Config.img_size).to(Config.device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.num_classes,
    ), f"Model output shape mismatch. Expected (2, {Config.num_classes}), got {output.shape}"
    print("Model architecture verification passed.")

    # 5. Verify Loss Function
    print("\n[5] Verifying WeightedSoftTargetCrossEntropy Loss...")
    criterion = WeightedSoftTargetCrossEntropy()

    # Dummy logits and targets
    dummy_logits = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=Config.device
    )
    dummy_targets = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=Config.device
    )

    loss = criterion(dummy_logits, dummy_targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"Loss computed successfully: {loss.item():.4f}")

    # 6. Verify Training Pipeline (Trainer)
    print("\n[6] Verifying Training Pipeline (running 1 fold)...")

    # Step 6a: Get Data and Folds
    # Note: trainer.get_folds caches to 'folds.parquet' in working_dir.
    # Since we changed working_dir, it will create a new one.
    df_full = trainer.get_all_data()
    df_folds = trainer.get_folds(df_full, n_folds=Config.n_folds, seed=Config.seed)

    assert "fold" in df_folds.columns, "Fold column missing in dataframe"
    assert (
        df_folds["fold"].nunique() == Config.n_folds
    ), f"Expected {Config.n_folds} folds, got {df_folds['fold'].nunique()}"

    # Step 6b: Calculate Weights
    class_weights = trainer.calculate_class_weights(df_folds, Config.device)
    assert class_weights.shape[0] == Config.num_classes, "Class weights shape mismatch"

    # Step 6c: Run a single fold (Fold 0)
    # This calls train_one_epoch, validate, and saves the model
    print("Running Fold 0...")
    best_auc = trainer.run_fold(0, df_folds, class_weights)

    expected_model_path = os.path.join(
        Config.checkpoint_dir, f"{Config.model_name}_fold_0.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model checkpoint not found at {expected_model_path}"
    assert 0 <= best_auc <= 1.0, f"AUC score {best_auc} out of range"
    print(f"Fold 0 completed. Best AUC: {best_auc:.4f}")

    # 7. Verify Inference Pipeline
    print("\n[7] Verifying Inference Pipeline...")

    # We need models for all folds to run the full ensemble inference as written in trainer.py
    # Since we only ran fold 0, let's duplicate the checkpoint for fold 1 to simulate a full run
    fold_1_path = os.path.join(Config.checkpoint_dir, f"{Config.model_name}_fold_1.pth")
    shutil.copy(expected_model_path, fold_1_path)

    # Run generation
    # We use the inference module directly to test that specific file's logic
    inference.generate_submission(
        test_metadata_path=Config.test_metadata_path,
        submission_path=Config.submission_path,
        checkpoint_dir=Config.checkpoint_dir,
        model_name=Config.model_name,
        n_folds=Config.n_folds,
        num_classes=Config.num_classes,
        target_cols=Config.target_cols,
        img_size=Config.img_size,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        device=Config.device,
        seed=Config.seed,
    )

    # Assertions on submission
    assert os.path.exists(Config.submission_path), "Submission file was not created"

    df_sub = pd.read_csv(Config.submission_path)
    assert len(df_sub) == len(
        df_test_subset
    ), f"Submission length mismatch. Expected {len(df_test_subset)}, got {len(df_sub)}"
    assert (
        list(df_sub.columns) == ["image_id"] + Config.target_cols
    ), "Submission columns mismatch"

    # Check probability constraints (rows should sum to ~1.0 if softmax was applied, though multi-label might differ,
    # here we used softmax in inference.py so they should sum to 1)
    row_sums = df_sub[Config.target_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("\n[8] Success! All components verified.")
    print(f"Demo artifacts stored in: {DEMO_DIR}")
    print(f"Submission head:\n{df_sub.head()}")


if __name__ == "__main__":
    main()
