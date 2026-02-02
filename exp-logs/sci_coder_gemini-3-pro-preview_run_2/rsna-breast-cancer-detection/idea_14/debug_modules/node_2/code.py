import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.utils import set_seed, get_device, probabilistic_f1
from library.data import get_dataloaders
from library.model import DSGEHNModel
from library.train import train_one_epoch, validate


def main():
    # =========================================================================
    # 1. Configuration for Demo
    # =========================================================================
    print("Setting up demo configuration...")
    # Override Config to run a fast demo in a separate directory
    Config.WORK_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = Config.WORK_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.PRETRAINED = False  # Disable downloading weights for speed

    # Ensure directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Loading & Subsetting
    # =========================================================================
    print("Loading data...")
    # Load full loaders first. This triggers metadata processing.
    # We set load_cached_data=False to force regeneration and avoid loading
    # stale/incompatible artifacts from previous runs. Cite debug_lesson_3.
    full_train_loader, full_val_loader, full_test_loader, feature_meta = (
        get_dataloaders(
            load_cached_data=False,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )
    )

    print("Creating data subsets for rapid demonstration...")
    # Create tiny subsets (12 samples = 3 batches of 4)
    subset_size = 12

    train_dataset = full_train_loader.dataset
    val_dataset = full_val_loader.dataset
    test_dataset = full_test_loader.dataset

    # Ensure we don't exceed dataset size
    train_subset = Subset(train_dataset, range(min(len(train_dataset), subset_size)))
    val_subset = Subset(val_dataset, range(min(len(val_dataset), subset_size)))
    test_subset = Subset(test_dataset, range(min(len(test_dataset), subset_size)))

    # Create new loaders for the subsets
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify Data Batch Structure
    print("Verifying data batch structure...")
    batch = next(iter(train_loader))
    assert "image" in batch, "Batch missing 'image' key"
    assert "categorical" in batch, "Batch missing 'categorical' key"
    assert "continuous" in batch, "Batch missing 'continuous' key"
    assert "label" in batch, "Batch missing 'label' key"

    # Check dimensions: (Batch, Channels, Height, Width)
    # Config.IMG_SIZE is (640, 640). Input is 1 channel.
    expected_shape = (Config.BATCH_SIZE, 1, 640, 640)
    assert (
        batch["image"].shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {batch['image'].shape}"

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing DSGEHNModel...")
    model = DSGEHNModel(feature_meta, pretrained=Config.PRETRAINED)
    model.to(device)

    # Verify Forward Pass
    print("Verifying model forward pass...")
    imgs = batch["image"].to(device)
    cats = batch["categorical"].to(device)
    conts = batch["continuous"].to(device)

    # Model returns (final_logits, aux_logits)
    final_logits, aux_logits = model(imgs, cats, conts)

    assert final_logits.shape == (Config.BATCH_SIZE, 1), "Final logits shape mismatch"
    assert aux_logits.shape == (Config.BATCH_SIZE, 1), "Aux logits shape mismatch"
    print("Forward pass successful.")

    # =========================================================================
    # 4. Training Loop Demonstration
    # =========================================================================
    print("Running training loop demonstration...")

    # Setup optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )
    pos_weight = torch.tensor(Config.POS_WEIGHT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train 1 Epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, 0, scheduler
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    val_loss, val_pf1 = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val pF1: {val_pf1:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_pf1 <= 1.0, "pF1 score out of valid range [0, 1]"

    # =========================================================================
    # 5. Metric Verification
    # =========================================================================
    print("Verifying probabilistic F1 logic...")
    # Manual test case
    y_true_demo = np.array([1, 0, 1, 0])
    y_pred_demo = np.array([0.9, 0.1, 0.8, 0.2])
    # pTP = 0.9 + 0.8 = 1.7
    # pFP = 0.1 + 0.2 = 0.3
    # pPrecision = 1.7 / (1.7 + 0.3) = 0.85
    # pRecall = 1.7 / 2.0 = 0.85
    # pF1 = 0.85
    score = probabilistic_f1(y_true_demo, y_pred_demo)
    assert abs(score - 0.85) < 1e-5, f"Metric calculation mismatch: {score}"

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("Running inference demonstration...")
    model.eval()
    all_preds = []

    # Get the dataframe subset corresponding to our test subset indices
    # We access the original dataframe from the dataset
    test_df_subset = test_dataset.df.iloc[test_subset.indices].copy()

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            cats = batch["categorical"].to(device)
            conts = batch["continuous"].to(device)

            logits, _ = model(imgs, cats, conts)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)

    # Assign predictions
    test_df_subset["cancer"] = all_preds

    # Aggregate by prediction_id (Max probability per patient/laterality group)
    submission = test_df_subset.groupby("prediction_id")["cancer"].max().reset_index()

    print("Saving submission...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "prediction_id" in loaded_sub.columns, "Submission missing prediction_id"
    assert "cancer" in loaded_sub.columns, "Submission missing cancer column"
    assert len(loaded_sub) > 0, "Submission file is empty"

    print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(loaded_sub)} rows.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
