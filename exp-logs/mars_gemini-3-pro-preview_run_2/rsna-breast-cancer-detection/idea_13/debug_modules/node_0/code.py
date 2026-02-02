import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import from provided library files
from library.config import Config
from library.dataset import BreastCancerDataset
from library.model import TriSpectralHybridModel
from library.engine import fit, set_seed, pf1_score
from library.image_utils import generate_tri_spectral_tensor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Breast Cancer Detection Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for the demo to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute requirements for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for demo
    Config.NUM_WORKERS = 2
    Config.IMAGE_SIZE = 256  # Smaller image size for faster processing in demo

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # =========================================================================
    # 2. Logic Verification (Unit Tests)
    # =========================================================================
    print("\n--- Verifying Logic ---")

    # A. Verify pF1 Score
    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = pf1_score(y_true, y_pred_perfect)
    assert np.isclose(score_perfect, 1.0), f"pF1 should be 1.0, got {score_perfect}"

    # Case 2: Zero prediction
    y_pred_zero = np.array([0.0, 0.0, 0.0, 0.0])
    score_zero = pf1_score(y_true, y_pred_zero)
    # pRecall will be 0, so pF1 should be 0
    assert np.isclose(score_zero, 0.0), f"pF1 should be 0.0, got {score_zero}"

    print("pF1 Score Logic: Verified.")

    # B. Verify Image Processing Pipeline
    # We grab a real file path from the metadata to test
    train_meta_path = Config.TRAIN_METADATA_PATH
    if os.path.exists(train_meta_path):
        df_temp = pd.read_csv(train_meta_path)
        if not df_temp.empty:
            # Construct full path
            rel_path = df_temp.iloc[0]["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Generate tensor
            tensor = generate_tri_spectral_tensor(
                full_path,
                size=Config.IMAGE_SIZE,
                gamma=Config.GAMMA_VALUE,
                clahe_clip=Config.CLAHE_CLIP_LIMIT,
                clahe_grid=Config.CLAHE_TILE_GRID_SIZE,
            )

            # Checks
            assert isinstance(tensor, np.ndarray), "Output must be numpy array"
            assert tensor.shape == (
                Config.IMAGE_SIZE,
                Config.IMAGE_SIZE,
                3,
            ), f"Shape mismatch: {tensor.shape}"
            assert tensor.dtype == np.float32, "Dtype must be float32"
            assert (
                tensor.min() >= 0.0 and tensor.max() <= 1.0
            ), "Values must be normalized [0, 1]"

            print("Image Processing Pipeline: Verified.")
    else:
        print(
            "Warning: Train metadata not found, skipping image pipeline verification."
        )

    # =========================================================================
    # 3. Data Loading (Subsets)
    # =========================================================================
    print("\n--- Initializing Datasets ---")

    # Load full datasets
    full_train_ds = BreastCancerDataset(mode="train")
    full_val_ds = BreastCancerDataset(mode="val")

    print(f"Full Train Size: {len(full_train_ds)}")
    print(f"Full Val Size: {len(full_val_ds)}")

    # Create small subsets for rapid demo execution
    # We take 20 samples for training and 10 for validation
    train_indices = range(min(20, len(full_train_ds)))
    val_indices = range(min(10, len(full_val_ds)))

    train_subset = Subset(full_train_ds, train_indices)
    val_subset = Subset(full_val_ds, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"Subset Train Batches: {len(train_loader)}")

    # =========================================================================
    # 4. Model Initialization & Check
    # =========================================================================
    print("\n--- Initializing Model ---")

    model = TriSpectralHybridModel()
    model.to(Config.DEVICE)

    # Verify Forward Pass
    dummy_img = torch.randn(
        2, Config.NUM_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(Config.DEVICE)
    dummy_tab = torch.randn(2, 10).to(Config.DEVICE)  # 10 tabular features

    with torch.no_grad():
        output = model(dummy_img, dummy_tab)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model Forward Pass: Verified.")

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    print("\n--- Starting Training Demo ---")

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler setup for 1 epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    # Run Training
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
    )

    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("\n--- Generating Submission ---")

    # Load Test Data
    test_ds = BreastCancerDataset(mode="test")
    # Use a subset if test is large, but usually test.csv provided is small/partial
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Load Best Model
    model.load_state_dict(
        torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    predictions = []
    prediction_ids = []

    # The dataset df has the prediction_ids
    # We can iterate the loader for predictions and use the dataset dataframe for IDs
    # Note: DataLoader order is preserved because shuffle=False

    with torch.no_grad():
        for images, tabular, _ in test_loader:
            images = images.to(Config.DEVICE)
            tabular = tabular.to(Config.DEVICE)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)

    # Get IDs from the dataframe directly
    # Ensure the length matches
    ids = test_ds.df["prediction_id"].values

    # If we used a subset for test, we would need to slice IDs.
    # Here we processed the whole available test set.
    assert len(predictions) == len(ids), "Mismatch between predictions and IDs"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"prediction_id": ids, "cancer": predictions})

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    print(submission_df.head())

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
