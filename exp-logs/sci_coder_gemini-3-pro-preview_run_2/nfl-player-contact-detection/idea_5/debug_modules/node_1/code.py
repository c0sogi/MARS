import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.losses import FocalLoss
from library.architecture import CFTCN
from library.data_loader import prepare_data, NFLDataset
from library.engine import fit, inference


def setup_demo_environment():
    """
    Sets up a demo environment by creating a subset of the metadata
    to ensure the pipeline runs quickly.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_base = "./working/demo_run"
    demo_meta = os.path.join(demo_base, "metadata")
    demo_working = os.path.join(demo_base, "working")
    demo_submission = os.path.join(demo_base, "submission")

    os.makedirs(demo_meta, exist_ok=True)
    os.makedirs(demo_working, exist_ok=True)
    os.makedirs(demo_submission, exist_ok=True)

    # Override Config to point to demo directories
    Config.METADATA_DIR = demo_meta
    Config.WORKING_DIR = demo_working
    Config.SUBMISSION_DIR = demo_submission

    # Override Training Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Create subset of metadata (1 game_play per split)
    # We read from the original read-only metadata
    original_meta_dir = "./metadata"

    for split in ["train", "validation", "test"]:
        original_path = os.path.join(original_meta_dir, f"{split}.csv")
        df = pd.read_csv(original_path)

        # Get the first unique game_play
        first_gp = df["game_play"].unique()[0]
        subset_df = df[df["game_play"] == first_gp].copy()

        # Save to demo metadata directory
        save_path = os.path.join(demo_meta, f"{split}.csv")
        subset_df.to_csv(save_path, index=False)
        print(
            f"Created {split} subset with {len(subset_df)} rows (GamePlay: {first_gp})"
        )

    return demo_meta


def verify_model_architecture(device):
    """
    Instantiates the model and runs a dummy input to verify shapes.
    """
    print("\nVerifying model architecture...")
    model = CFTCN().to(device)

    # Input shape: (Batch, INPUT_WIDTH)
    # INPUT_WIDTH = NUM_FEATURES_PER_STEP * WINDOW_SIZE
    input_dim = Config.INPUT_WIDTH
    batch_size = 4

    dummy_input = torch.randn(batch_size, input_dim).to(device)
    output = model(dummy_input)

    # Expected output: (Batch, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"

    print("Model architecture verification passed.")
    return model


def run_pipeline():
    # 1. Set Seeds for Reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Setup Demo Data
    setup_demo_environment()

    # 3. Data Processing & Loading
    # Note: load_cached_data=False forces regeneration from our new subset metadata
    print("\nPreparing Training Data...")
    train_dataset = prepare_data("train", load_cached_data=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    print("Preparing Validation Data...")
    val_dataset = prepare_data("validation", load_cached_data=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Data
    sample_x, sample_y = train_dataset[0]
    assert (
        sample_x.shape[0] == Config.INPUT_WIDTH
    ), "Feature dimension mismatch in dataset."
    assert not np.isnan(sample_x).any(), "NaNs detected in training features."
    print(
        f"Data loaded successfully. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}"
    )

    # 4. Model & Training Setup
    model = verify_model_architecture(device)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    print("\nStarting Training Loop...")
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    trained_model, best_mcc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=model_save_path,
    )

    print(f"Training complete. Best Validation MCC: {best_mcc:.4f}")

    # Verify model file exists
    assert os.path.exists(model_save_path), "Best model file was not saved."

    # 6. Inference
    print("\nRunning Inference...")
    # Load test metadata dataframe for submission alignment
    test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
    df_test_meta = pd.read_csv(test_meta_path)

    # Prepare test dataset
    test_dataset = prepare_data("test", load_cached_data=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run inference using the best threshold found during training (or a default)
    # Since fit() returns the best model state, we can use it directly.
    # We'll use a fixed threshold for demo purposes, or we could track the best thresh from fit.
    inference_threshold = 0.5

    inference(
        model=trained_model,
        test_loader=test_loader,
        test_df=df_test_meta,
        device=device,
        threshold=inference_threshold,
    )

    # 7. Verify Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found."

    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == len(df_test_meta), "Submission row count mismatch."
    assert (
        "contact_id" in df_sub.columns and "contact" in df_sub.columns
    ), "Submission columns missing."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary predictions."

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    run_pipeline()
