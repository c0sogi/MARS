import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import Config
import library.config as lib_config
from library.data_loader import get_dataloaders
from library.model import get_model, predict
from library.trainer import run_training, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("============================================================")
    print("   Library Demonstration: Deep Parallel Vector-DCN-ResNet   ")
    print("============================================================")

    # 1. Setup Environment & Reproducibility
    # ------------------------------------------------------------
    set_seed(42)

    # Define a separate working directory for this demo to avoid cache conflicts
    DEMO_DIR = "./working/demo_execution"
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")

    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    print(f"Working directory: {DEMO_DIR}")

    # 2. Create Mini-Dataset (Optimize for Speed)
    # ------------------------------------------------------------
    print("\n[Step 1] Creating Mini-Dataset for rapid demonstration...")

    # Paths to original metadata
    orig_train_path = "./metadata/train.parquet"
    orig_val_path = "./metadata/val.parquet"
    orig_test_path = "./metadata/test.parquet"

    # Paths for mini datasets
    mini_train_path = os.path.join(DEMO_DIR, "train.parquet")
    mini_val_path = os.path.join(DEMO_DIR, "val.parquet")
    mini_test_path = os.path.join(DEMO_DIR, "test.parquet")

    # Load only the first 2000 rows to ensure speed
    # We use pandas to read and write parquet
    pd.read_parquet(orig_train_path).head(2000).to_parquet(mini_train_path, index=False)
    pd.read_parquet(orig_val_path).head(1000).to_parquet(mini_val_path, index=False)
    pd.read_parquet(orig_test_path).head(1000).to_parquet(mini_test_path, index=False)

    print("Mini-datasets created successfully.")

    # 3. Override Configuration
    # ------------------------------------------------------------
    print("\n[Step 2] Configuring pipeline parameters...")

    # Monkey-patch the Config class to use our mini-dataset and fast training settings
    Config.METADATA_DIR = DEMO_DIR
    Config.TRAIN_DATA = mini_train_path
    Config.VAL_DATA = mini_val_path
    Config.TEST_DATA = mini_test_path
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = DEMO_CACHE_DIR

    # Reduce hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.HIDDEN_DIM = 64  # Smaller dimension for demo
    Config.RESNET_BLOCKS = 1
    Config.DCN_LAYERS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    print(f"Config updated: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 4. Data Loading & Processing
    # ------------------------------------------------------------
    print("\n[Step 3] Loading and processing data...")

    # Force processing from scratch (load_cached_data=False) to demonstrate the pipeline
    train_loader, val_loader, test_loader, num_features, num_classes, test_ids = (
        get_dataloaders(load_cached_data=False)
    )

    # Validation
    print(f"Data Loaded: Features={num_features}, Classes={num_classes}")
    print(f"Train Batches: {len(train_loader)}")

    # Assertions to verify data integrity
    assert num_features > 0, "Number of features must be positive."
    assert num_classes > 1, "Number of classes must be > 1 for classification."
    assert len(test_ids) == 1000, f"Expected 1000 test IDs, got {len(test_ids)}"

    # Check batch shape
    sample_batch, sample_labels = next(iter(train_loader))
    assert sample_batch.shape[1] == num_features, "Batch feature dimension mismatch."
    assert (
        sample_labels.shape[0] == sample_batch.shape[0]
    ), "Batch label count mismatch."
    print("Data integrity checks passed.")

    # 5. Model Initialization
    # ------------------------------------------------------------
    print("\n[Step 4] Initializing Deep Parallel Vector-DCN-ResNet...")

    device = Config.DEVICE
    model = get_model(input_dim=num_features, num_classes=num_classes).to(device)

    # Validation: Forward pass
    dummy_input = torch.randn(2, num_features).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        2,
        num_classes,
    ), f"Model output shape mismatch. Expected (2, {num_classes}), got {dummy_output.shape}"
    print("Model initialized and forward pass verified.")

    # 6. Training Loop
    # ------------------------------------------------------------
    print("\n[Step 5] Running training loop...")

    # Use the trainer library to run training
    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        device=device,
    )

    # Validation: Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not created."
    print(f"Training complete. Best model saved to {best_model_path}")

    # 7. Inference & Submission
    # ------------------------------------------------------------
    print("\n[Step 6] Generating predictions on test set...")

    # Generate raw predictions (indices)
    raw_preds = predict(trained_model, test_loader, device)

    assert len(raw_preds) == len(
        test_ids
    ), "Prediction count does not match Test ID count."

    # Inverse Transform Labels
    # We need to load the label encoder cached during processing
    meta_path = os.path.join(Config.CACHE_DIR, "metadata.npy")
    meta_dict = np.load(meta_path, allow_pickle=True).item()
    le = meta_dict["label_encoder"]

    final_preds = le.inverse_transform(raw_preds)

    # Create Submission DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    print("Sample Submission:")
    print(submission.head())

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file not created."
    print(f"Submission saved to {submission_path}")

    print("\n============================================================")
    print("   Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    main()
