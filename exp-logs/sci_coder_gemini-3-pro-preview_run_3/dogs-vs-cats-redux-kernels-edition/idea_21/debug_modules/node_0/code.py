import os
import shutil
import torch
import pandas as pd
import numpy as np
import logging

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataframes, get_dataloader, CatDogDataset
from library.modeling import get_model
from library.engine import train_one_epoch, validate, predict
from library.calibration import Calibrator
from library.pipeline import run_kfold_ensemble

# Configure logger to be less verbose for the demo
logging.getLogger("train").setLevel(logging.WARNING)
logging.getLogger("modeling").setLevel(logging.WARNING)
logging.getLogger("pipeline").setLevel(logging.INFO)
logging.getLogger("calibration").setLevel(logging.INFO)


def setup_demo_environment():
    """
    Sets up a lightweight environment for demonstration:
    1. Creates a temporary working directory.
    2. Generates mini-datasets from the original metadata.
    3. Overrides Config parameters for speed (1 model, 1 epoch, small batch).
    """
    print(">>> Setting up demo environment...")
    seed_everything(42)

    # Create a separate working directory for the demo
    demo_working_dir = "./working/demo_run_script"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir)

    # Update Config to use this directory
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # --- Create Mini Datasets ---
    # Read original metadata
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Sample small subsets (e.g., 50 train, 20 val, 20 test)
    # Ensure we have both classes in train/val
    mini_train = (
        pd.concat(
            [
                train_full[train_full["label"] == 0].head(25),
                train_full[train_full["label"] == 1].head(25),
            ]
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    mini_val = (
        pd.concat(
            [
                val_full[val_full["label"] == 0].head(10),
                val_full[val_full["label"] == 1].head(10),
            ]
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    mini_test = test_full.head(20).reset_index(drop=True)

    # Save mini metadata
    mini_train_path = os.path.join(demo_working_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_working_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_working_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # --- Override Config ---
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    # Use only one model, small resolution, 1 epoch
    Config.MODELS = {
        "resnet50": {
            "backbone": "resnet50.a1_in1k",
            "batch_size": 8,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "phases": [
                {"img_size": 128, "epochs": 1},
            ],
        }
    }
    Config.N_FOLDS = 2

    print(">>> Environment setup complete. Using mini datasets.")


def demo_data_loading():
    """
    Demonstrates loading dataframes and creating a DataLoader.
    """
    print("\n>>> Demonstrating Data Loading...")

    # Load dataframes (will use the mini csvs set in Config)
    # load_cached_data=False ensures we read the new CSVs
    train_df, val_df, test_df = get_dataframes(load_cached_data=False)

    assert len(train_df) == 50, f"Expected 50 train samples, got {len(train_df)}"
    assert len(val_df) == 20, f"Expected 20 val samples, got {len(val_df)}"

    # Create DataLoader
    # Use 'resnet50' config which we set to batch_size 8, img_size 128
    loader = get_dataloader(train_df, img_size=128, batch_size=8, mode="train")

    # Fetch one batch
    images, labels = next(iter(loader))

    # Verification
    print(f"Batch Shapes - Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == (8, 3, 128, 128), "Incorrect image tensor shape"
    assert labels.shape == (8,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"

    print("Data Loading verification passed.")
    return loader


def demo_modeling():
    """
    Demonstrates model instantiation.
    """
    print("\n>>> Demonstrating Model Instantiation...")

    model = get_model("resnet50", pretrained=False)  # False for speed
    model.to(Config.DEVICE)

    # Test Forward Pass
    dummy_input = torch.randn(2, 3, 128, 128).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model output should be [Batch, 1]"

    print("Modeling verification passed.")
    return model


def demo_engine(model, loader):
    """
    Demonstrates training and validation steps using the engine.
    """
    print("\n>>> Demonstrating Engine (Train/Val/Predict)...")

    device = torch.device(Config.DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Train One Epoch (Limit to 2 steps for speed)
    print("Running training step...")
    loss = train_one_epoch(
        model, loader, optimizer, criterion, device, scaler, steps_per_epoch=2
    )
    print(f"Training Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # Validation (Limit to 2 steps)
    print("Running validation step...")
    val_loss, preds, targets = validate(
        model, loader, criterion, device, steps_per_epoch=2
    )
    print(f"Validation Loss: {val_loss:.4f}")
    assert len(preds) > 0, "No predictions returned"
    assert len(preds) == len(targets), "Predictions and targets length mismatch"

    print("Engine verification passed.")


def demo_calibration():
    """
    Demonstrates the Calibrator class.
    """
    print("\n>>> Demonstrating Calibration...")

    # Synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    # Generate predictions correlated with truth
    y_pred_raw = np.random.rand(100)
    y_pred_raw = np.where(y_true == 1, y_pred_raw + 0.2, y_pred_raw - 0.2)
    y_pred_raw = np.clip(y_pred_raw, 0.01, 0.99)

    calibrator = Calibrator(method="isotonic")
    calibrator.fit(y_pred_raw, y_true)

    # Transform new data
    y_test = np.random.rand(10)
    y_calibrated = calibrator.transform(y_test)

    print(
        f"Calibrated Output Range: [{y_calibrated.min():.4f}, {y_calibrated.max():.4f}]"
    )
    assert (
        y_calibrated.min() >= 0.0 and y_calibrated.max() <= 1.0
    ), "Calibration output out of bounds"
    assert y_calibrated.shape == y_test.shape, "Calibration output shape mismatch"

    print("Calibration verification passed.")


def demo_pipeline():
    """
    Demonstrates the full pipeline execution (K-Fold, Training, Inference, Submission).
    """
    print("\n>>> Demonstrating Full Pipeline Execution...")

    # run_kfold_ensemble uses the global Config we modified in setup_demo_environment.
    # debug=True ensures reduced folds/epochs are enforced (though we already set them).
    run_kfold_ensemble(debug=True)

    # Verify submission file
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file not generated"

    df = pd.read_csv(sub_path)
    print(f"Submission File: {len(df)} rows")
    print(df.head())

    # Check against mini_test length (20)
    assert (
        len(df) == 20
    ), f"Submission length {len(df)} does not match test set size (20)"
    assert "id" in df.columns and "label" in df.columns, "Submission columns missing"

    print("Pipeline verification passed.")


if __name__ == "__main__":
    try:
        setup_demo_environment()

        # Run individual component demonstrations
        loader = demo_data_loading()
        model = demo_modeling()
        demo_engine(model, loader)
        demo_calibration()

        # Clean up memory before full pipeline
        del model, loader
        torch.cuda.empty_cache()

        # Run full pipeline demonstration
        demo_pipeline()

        print("\n>>> All demonstrations completed successfully.")

    except Exception as e:
        print(f"\n!!! Demonstration Failed: {e}")
        raise e
