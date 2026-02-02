import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, scale_target
from library.dataset import load_metadata_splits, PawpularityDataset, get_transforms
from library.model_factory import PetModel
from library.trainer import fit, predict, save_submission
from library.stacking import StackingTrainer


def run_demo_pipeline():
    print("Initializing Demo Pipeline...")

    # =========================================================================
    # 1. Configuration Overrides for Speed/Demo
    # =========================================================================
    # Modify Config globally to run a fast demo
    Config.debug = True
    Config.debug_sample_size = 50  # Small subset for speed
    Config.epochs = 1
    Config.batch_size = 8
    Config.n_folds = 2

    # Use a lightweight model available in timm for demonstration
    # We override the dictionary to just one model to simulate a single-model stack for the demo
    Config.models = {"demo_model": "resnet18"}

    # Redirect outputs to a working directory
    Config.output_dir = "./working/demo_run"
    Config.submission_dir = "./working/demo_run"
    Config.submission_path = os.path.join(Config.submission_dir, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)

    print(f"Configuration set. Output dir: {Config.output_dir}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading data...")
    train_df, val_df, test_df = load_metadata_splits(
        debug=Config.debug, sample_size=Config.debug_sample_size
    )

    # Verify data loading
    assert len(train_df) == Config.debug_sample_size
    assert len(val_df) == Config.debug_sample_size
    assert len(test_df) == Config.debug_sample_size

    # Create Datasets
    train_ds = PawpularityDataset(train_df, transforms=get_transforms("train"))
    val_ds = PawpularityDataset(val_df, transforms=get_transforms("valid"))
    test_ds = PawpularityDataset(test_df, transforms=get_transforms("test"), test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Data loaders ready.")

    # =========================================================================
    # 3. Base Model Training
    # =========================================================================
    print("Training Base Model (ResNet18)...")

    model_name = "demo_model"
    model_save_path = Config().get_model_path(
        model_name, fold=0
    )  # Using fold 0 for demo

    # Instantiate Model
    model = PetModel(model_name, pretrained=True)
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.head_lr)
    scheduler = None  # Skip scheduler for 1 epoch demo

    # Train (Fit)
    # fit() returns the scaled predictions [0, 1] for the validation set (OOF)
    oof_preds_scaled = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        Config.device,
        epochs=Config.epochs,
        save_path=model_save_path,
        patience=1,
    )

    # Verify OOF predictions
    assert len(oof_preds_scaled) == len(val_df)
    assert oof_preds_scaled.min() >= 0.0 and oof_preds_scaled.max() <= 1.0

    # Save OOF predictions to disk
    # The StackingTrainer expects a CSV with Id and the prediction column
    oof_df = val_df[["Id"]].copy()
    # Note: StackingTrainer usually expects the column name to match the model key or be 'Pawpularity'
    oof_df[model_name] = oof_preds_scaled

    oof_path = Config().get_oof_path(model_name)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

    # =========================================================================
    # 4. Stacking (Meta-Learner)
    # =========================================================================
    print("Running Stacking Trainer...")

    # HACK: The StackingTrainer loads 'Config.train_metadata_path' to get ground truth.
    # Since we generated OOFs for the *validation* set (val_df), we must tell
    # the StackingTrainer to treat the validation file as the source of truth for these IDs.
    original_train_path = Config.train_metadata_path
    Config.train_metadata_path = Config.val_metadata_path

    try:
        stacker = StackingTrainer()

        # Load OOF data (X) and Ground Truth (y)
        # We disable caching to ensure it reads our newly generated file
        X_meta, y_meta = stacker.load_oof_data(load_cached_data=False)

        # Verify shapes
        # X_meta shape: (n_samples, n_models) -> (50, 1)
        assert X_meta.shape == (len(val_df), 1)

        # Train Meta-Learner
        stacker.train(X_meta, y_meta)

    finally:
        # Restore config path just in case
        Config.train_metadata_path = original_train_path

    # =========================================================================
    # 5. Inference & Submission
    # =========================================================================
    print("Generating Test Predictions...")

    # 1. Base Model Inference
    # predict() returns UNSCALED predictions [1, 100]
    raw_test_preds = predict(model, test_loader, Config.device)

    # 2. Prepare for Stacking
    # StackingTrainer expects SCALED predictions [0, 1] in a dictionary
    scaled_test_preds = scale_target(raw_test_preds)
    test_preds_map = {model_name: scaled_test_preds}

    # 3. Meta-Learner Inference
    final_preds = stacker.predict(test_preds_map)

    # Verify final predictions
    assert len(final_preds) == len(test_df)
    assert final_preds.min() >= 1.0 and final_preds.max() <= 100.0

    # 4. Save Submission
    save_submission(test_df["Id"].values, final_preds, Config.submission_path)

    # Verify file creation
    assert os.path.exists(Config.submission_path)

    # Check content
    sub_df = pd.read_csv(Config.submission_path)
    print("\nSample Submission:")
    print(sub_df.head())

    print("\nDemo Pipeline Completed Successfully.")


if __name__ == "__main__":
    run_demo_pipeline()
