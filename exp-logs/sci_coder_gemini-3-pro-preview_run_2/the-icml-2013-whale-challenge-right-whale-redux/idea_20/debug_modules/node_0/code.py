import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.data import get_dataloaders
from library.models import WhaleClassifier
from library.engine import Trainer
from library.pseudo_labeling import infer_on_test, generate_pseudo_labels
from library.ensemble import (
    load_oof_features,
    load_test_features,
    train_meta_learner,
    generate_submission,
)


def create_mini_metadata(config, num_samples=20):
    """
    Creates small subsets of the original metadata for rapid demonstration.
    """
    print(f"\n[Demo] Creating mini metadata with {num_samples} samples...")

    # Load original metadata
    train_df = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

    # Sample subsets
    mini_train = train_df.head(num_samples).copy()
    mini_val = val_df.head(num_samples).copy()
    mini_test = test_df.head(num_samples).copy()

    # Define paths for mini metadata
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    # Save to disk
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update config to point to mini metadata
    config.TRAIN_CSV = mini_train_path
    config.VAL_CSV = mini_val_path
    config.TEST_CSV = mini_test_path

    print(f"[Demo] Mini metadata saved to {config.WORKING_DIR}")
    return len(mini_train), len(mini_val), len(mini_test)


def simulate_oof_and_preds(config, model_name, num_train_total, num_test_total):
    """
    Simulates OOF and Test prediction files that would normally be generated
    by training K folds. This allows us to demonstrate the ensemble logic.
    """
    print(f"\n[Demo] Simulating OOF and Prediction files for {model_name}...")

    # Create directories
    os.makedirs(config.OOF_DIR, exist_ok=True)
    os.makedirs(config.PREDS_DIR, exist_ok=True)

    # We need to simulate predictions for every fold
    # In the library logic, OOFs are loaded based on validation indices of the fold.
    # To simplify, we'll just generate random arrays of the correct size.
    # Note: load_oof_features expects specific lengths based on the split.
    # Since we are using a mini-dataset, we need to be careful with lengths if we want exact matching.
    # However, for this demo, we will rely on the fact that we are mocking the files.

    # Actually, load_oof_features iterates through folds and expects the file to contain
    # preds for that fold's validation set.

    # Let's reconstruct the split to get exact sizes
    from sklearn.model_selection import StratifiedKFold

    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    y = full_df["label"].values

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    for fold, (_, val_idx) in enumerate(skf.split(full_df, y)):
        n_val = len(val_idx)

        # Simulate OOFs (Best AUC and Best Loss)
        for metric in ["best_auc", "best_loss"]:
            fname = f"{model_name}_{metric}_fold_{fold}.npy"
            dummy_preds = np.random.rand(n_val).astype(np.float32)
            np.save(os.path.join(config.OOF_DIR, fname), dummy_preds)

        # Simulate Test Preds
        for metric in ["best_auc", "best_loss"]:
            fname = f"{model_name}_{metric}_fold_{fold}.npy"
            dummy_test_preds = np.random.rand(num_test_total).astype(np.float32)
            np.save(os.path.join(config.PREDS_DIR, fname), dummy_test_preds)


def main():
    # 1. Configuration
    print("--- Step 1: Configuration ---")
    # Initialize Config with debug parameters for speed
    config = Config(debug=True, num_epochs=1, batch_size=4)

    # Override paths to use a demo working directory
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
    config.OOF_DIR = os.path.join(config.WORKING_DIR, "oof")
    config.PREDS_DIR = os.path.join(config.WORKING_DIR, "preds")
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")
    config.SUBMISSION_FILE = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce complexity for demo
    config.NUM_FOLDS = 2
    config.MODELS = ["tf_efficientnet_b0_ns"]  # Use just one model for demo

    config.create_directories()
    seed_everything(config.SEED)

    # Logger
    logger = setup_logger(os.path.join(config.WORKING_DIR, "train.log"))
    logger.info("Starting Demo Execution...")

    # 2. Data Subsampling
    print("--- Step 2: Data Subsampling ---")
    n_train, n_val, n_test = create_mini_metadata(config, num_samples=50)

    # 3. Data Loading
    print("--- Step 3: Data Loading ---")
    # Get dataloaders for Fold 0
    train_loader, val_loader = get_dataloaders(
        config,
        fold=0,
        mode="train",
        load_cached_data=False,  # Force re-cache for mini data
    )

    # Validation: Check batch structure
    images, targets = next(iter(train_loader))
    print(f"Train Batch Shape: {images.shape}")
    print(f"Targets Shape: {targets.shape}")

    assert images.shape == (
        config.BATCH_SIZE,
        config.IN_CHANNELS,
        config.IMG_SIZE[0],
        config.IMG_SIZE[1],
    ), f"Unexpected image shape: {images.shape}"
    assert targets.shape[0] == config.BATCH_SIZE, "Target batch size mismatch"

    # 4. Model Initialization
    print("--- Step 4: Model Initialization ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.MODELS[0]

    model = WhaleClassifier(
        model_name=model_name,
        pretrained=False,  # False for speed/offline demo
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
    )
    model.to(device)

    # 5. Training Loop
    print("--- Step 5: Training ---")
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        fold=0,
        model_name=model_name,
        logger=logger,
    )

    trainer.fit()

    # Verify checkpoint creation
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_fold_0_best_auc.pth")
    if not os.path.exists(ckpt_path):
        # Fallback if AUC didn't improve (unlikely with init 0), check loss
        ckpt_path = os.path.join(
            config.CHECKPOINT_DIR, f"{model_name}_fold_0_best_loss.pth"
        )

    assert os.path.exists(ckpt_path), "Checkpoint was not created!"
    print(f"Checkpoint verified at {ckpt_path}")

    # 6. Inference
    print("--- Step 6: Inference ---")
    # Define checkpoints to use (List of tuples: (model_name, path))
    checkpoints = [(model_name, ckpt_path)]

    test_preds = infer_on_test(
        config, model_checkpoints=checkpoints, device=device, load_cached_preds=False
    )

    assert (
        len(test_preds) == n_test
    ), f"Prediction length mismatch: {len(test_preds)} vs {n_test}"
    print(f"Inference complete. Predictions shape: {test_preds.shape}")

    # 7. Pseudo-Labeling
    print("--- Step 7: Pseudo-Labeling ---")
    pseudo_df = generate_pseudo_labels(config, test_preds, conf_high=0.8, conf_low=0.2)
    print(f"Generated {len(pseudo_df)} pseudo-labels.")

    # 8. Ensemble & Submission
    print("--- Step 8: Ensemble & Submission ---")

    # To demonstrate the ensemble code, we need OOF and Pred files for all folds.
    # Since we only trained Fold 0, we will simulate the files for Fold 1 (and 0).
    total_train_samples = n_train + n_val  # Mini dataset total
    simulate_oof_and_preds(config, model_name, total_train_samples, n_test)

    # Load Features
    X_oof = load_oof_features(config, load_cached_data=False)
    X_test = load_test_features(config, load_cached_data=False)

    print(f"OOF Features Shape: {X_oof.shape}")
    print(f"Test Features Shape: {X_test.shape}")

    # Train Meta Learner
    meta_learner = train_meta_learner(X_oof)

    # Generate Submission
    # We need to point config.SAMPLE_SUBMISSION to a valid file matching our mini test set
    # Create a dummy sample submission for the mini test set
    mini_test_df = pd.read_csv(config.TEST_CSV)
    dummy_sample_sub = pd.DataFrame(
        {"clip": mini_test_df["clip"], "probability": [0] * len(mini_test_df)}
    )
    dummy_sample_sub_path = os.path.join(
        config.WORKING_DIR, "mini_sample_submission.csv"
    )
    dummy_sample_sub.to_csv(dummy_sample_sub_path, index=False)
    config.SAMPLE_SUBMISSION = dummy_sample_sub_path

    submission = generate_submission(config, meta_learner, X_test)

    assert len(submission) == n_test, "Submission length mismatch"
    assert (
        "clip" in submission.columns and "probability" in submission.columns
    ), "Submission columns missing"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
