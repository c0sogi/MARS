import os
import shutil
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import Config
from library.dataset import SIIMDataset
from library.model import MultiTaskFasterRCNN
from library.utils import collate_fn
import importlib
import library.engine
import library.submission

importlib.reload(library.engine)
importlib.reload(library.submission)

from library.engine import train_model, set_seed
from library.submission import generate_submission


def run_demonstration():
    print("=== Starting SIIM-FISABIO-RSNA Library Demonstration ===")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    # We override Config attributes to create a self-contained, fast demo.

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Point config paths to our demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Optimization for speed
    Config.IMG_SIZE = 320  # Smaller images for faster processing
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 2  # Moderate workers

    # Set global seed
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Configuration set. Output directory: {DEMO_DIR}")
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Dataset Loading & Verification
    # ---------------------------------------------------------
    print("\n--- Step 2: Dataset Loading ---")

    # Initialize Train Dataset (disable cache to force fresh load for demo)
    train_dataset = SIIMDataset(split="train", load_cached_data=False)

    # SUBSET: Keep only 20 samples
    train_dataset.df = train_dataset.df.iloc[:20].reset_index(drop=True)
    print(f"Train dataset initialized and subset to {len(train_dataset)} samples.")

    # Initialize Val Dataset
    val_dataset = SIIMDataset(split="val", load_cached_data=False)
    val_dataset.df = val_dataset.df.iloc[:10].reset_index(drop=True)
    print(f"Val dataset initialized and subset to {len(val_dataset)} samples.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Verify Batch Structure
    images, targets, image_ids = next(iter(train_loader))

    # Assertions
    assert len(images) == Config.BATCH_SIZE, "Batch size mismatch in images"
    assert len(targets) == Config.BATCH_SIZE, "Batch size mismatch in targets"

    # Check Image Tensor Shape: (3, H, W)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images[0].shape == expected_shape
    ), f"Image shape mismatch. Got {images[0].shape}, expected {expected_shape}"

    # Check Target Keys
    required_keys = {"boxes", "labels", "study_label", "image_id"}
    assert required_keys.issubset(
        targets[0].keys()
    ), f"Missing keys in target. Found: {targets[0].keys()}"

    # Check Study Label (should be scalar tensor)
    assert targets[0]["study_label"].dim() == 0, "Study label should be a scalar tensor"

    print("Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n--- Step 3: Model Initialization ---")

    model = MultiTaskFasterRCNN()
    model.to(device)

    # Prepare batch for device
    images_dev = [img.to(device) for img in images]
    targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Test Forward Pass (Training Mode)
    model.train()
    loss_dict = model(images_dev, targets_dev)

    print("Loss dictionary keys:", list(loss_dict.keys()))

    # Verify all required losses are present
    expected_losses = [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_study",
    ]
    for loss_name in expected_losses:
        assert loss_name in loss_dict, f"Missing loss component: {loss_name}"
        assert not torch.isnan(loss_dict[loss_name]), f"Loss {loss_name} is NaN"

    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n--- Step 4: Training Loop ---")

    # Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Verify Checkpoint
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print("Training complete. Checkpoint saved.")

    # ---------------------------------------------------------
    # 5. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n--- Step 5: Submission Generation ---")

    # Generate submission using the saved model and a subset of test data
    # num_samples=10 ensures we don't process the whole test set
    generate_submission(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, num_samples=10
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Verify Columns
    assert "Id" in df_sub.columns, "Submission missing 'Id' column"
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing 'PredictionString' column"

    # Verify Content (Should contain both study and image IDs)
    has_study_ids = df_sub["Id"].str.endswith("_study").any()
    has_image_ids = df_sub["Id"].str.endswith("_image").any()

    assert has_study_ids, "Submission lacks study-level predictions"
    assert has_image_ids, "Submission lacks image-level predictions"

    print("Submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
