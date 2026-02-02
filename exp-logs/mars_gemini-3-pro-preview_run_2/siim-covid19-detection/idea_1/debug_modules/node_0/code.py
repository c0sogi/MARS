import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, format_prediction_string
from library.dataset import SIIMDataset, get_transforms, collate_fn
from library.model import get_model
from library.engine import Engine


def run_demo():
    print("=== Starting Demonstration ===")

    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Speed
    print("Configuring for fast demonstration...")
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    # Define paths
    train_csv_path = Config.TRAIN_CSV
    val_csv_path = Config.VAL_CSV
    test_csv_path = Config.TEST_CSV

    # 2. Load Data and Create Subsets
    print("Loading metadata...")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Create tiny subsets for demonstration
    train_subset = train_df.head(10).reset_index(drop=True)
    val_subset = val_df.head(4).reset_index(drop=True)
    test_subset = test_df.head(4).reset_index(drop=True)

    print(
        f"Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # 3. Verify Dataset Logic
    print("Verifying Dataset logic...")
    train_dataset = SIIMDataset(
        train_subset, mode="train", transforms=get_transforms("train")
    )

    # Fetch one item to check integrity
    img, target, img_id = train_dataset[0]

    # Assertions
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert isinstance(target, dict), "Target should be a dictionary"
    assert "boxes" in target and "labels" in target, "Target missing keys"
    assert (
        target["boxes"].dim() == 2 and target["boxes"].shape[1] == 4
    ), "Boxes shape incorrect"

    print("Dataset verification passed.")

    # 4. Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_dataset = SIIMDataset(val_subset, mode="val", transforms=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 5. Initialize Model
    print("Initializing Model...")
    device = Config.DEVICE
    model = get_model(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 6. Train (Fit Model)
    print("Starting Training Loop (1 Epoch)...")
    engine = Engine(model, device, optimizer)

    # This will save 'best_model.pth' in Config.WORKING_DIR
    model_path = engine.fit_model(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify model was saved
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print("Training completed and model saved.")

    # 7. Inference
    print("Starting Inference...")
    test_dataset = SIIMDataset(
        test_subset, mode="test", transforms=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Run inference
    submission_df = engine.inference(test_loader, model_path)

    # 8. Verify Submission
    print("Verifying Submission...")
    expected_rows = len(test_subset) * 2  # One for study, one for image
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    assert (
        "id" in submission_df.columns and "PredictionString" in submission_df.columns
    ), "Submission columns mismatch"

    # Check format of a prediction string
    sample_pred = submission_df.iloc[0]["PredictionString"]
    assert isinstance(sample_pred, str), "PredictionString is not a string"
    # Basic check: should contain at least a label and confidence (e.g., 'negative 1' or 'none 1')
    assert len(sample_pred.split()) >= 2, "PredictionString format seems too short"

    print("Submission verification passed.")

    # 9. Verify Utility Helper
    print("Verifying Utility functions...")
    # Test format_prediction_string
    lbls = ["opacity", "opacity"]
    bxs = [[10, 10, 50, 50], [60, 60, 100, 100]]
    scrs = [0.95, 0.88]
    formatted = format_prediction_string(lbls, bxs, scrs)
    expected_substr = "opacity 0.950000 10 10 50 50 opacity 0.880000 60 60 100 100"
    assert formatted == expected_substr, f"Format helper failed. Got: {formatted}"
    print("Utility verification passed.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
