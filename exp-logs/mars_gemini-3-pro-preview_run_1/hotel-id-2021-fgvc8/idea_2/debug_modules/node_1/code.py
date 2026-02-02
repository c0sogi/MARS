import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.utils import Config, set_seed, get_device
from library.dataset import process_metadata, HotelDataset, get_transforms
from library.model import LightweightMetricModel
from library.trainer import train_one_epoch, validate, predict_and_submit, mapk


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration and environment...")

    # Override Config paths to use a demo directory within ./working
    # This prevents overwriting any main experiment files and keeps the demo self-contained.
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "checkpoint.pth")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Hyperparameters for speed
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 2

    # Create directories
    Config.setup_dirs()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Subsetting (Optimization for Speed)
    # -------------------------------------------------------------------------
    print("\n>>> Preparing DataLoaders (Subset)...")

    # Load metadata using the library function
    # We force load_cached_data=False to ensure we test the raw CSV reading logic
    train_df_full, val_df_full, test_df_full, encoder_classes = process_metadata(
        load_cached_data=False
    )

    print(f"Original Train size: {len(train_df_full)}")

    # SUBSETTING: Take only 64 samples for each split to ensure quick execution
    subset_size = 64
    train_df_subset = train_df_full.head(subset_size).copy()
    val_df_subset = val_df_full.head(subset_size).copy()
    test_df_subset = test_df_full.head(subset_size).copy()

    # Verify we have enough classes in the subset for the code to run without crashing
    # (Though ArcFace handles many classes, we just need to ensure labels are valid indices)
    # The label_idx in train_df_full was computed based on the full dataset, so indices are valid.

    # Create Datasets using the library class
    train_dataset = HotelDataset(
        df=train_df_subset, transform=get_transforms("train"), is_test=False
    )
    val_dataset = HotelDataset(
        df=val_df_subset, transform=get_transforms("val"), is_test=False
    )
    test_dataset = HotelDataset(
        df=test_df_subset, transform=get_transforms("test"), is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Assertion: Check DataLoader functionality
    images, labels = next(iter(train_loader))
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {labels.shape}"
    print("DataLoaders initialized and verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")

    num_classes = len(encoder_classes)
    model = LightweightMetricModel(
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        backbone_name=Config.BACKBONE,
        pretrained=False,  # False for speed, we don't need accurate weights for a demo
    )
    model = model.to(device)

    # Assertion: Check Forward Pass (Training Mode)
    # Expects ArcFace logits
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 1, 0, 1]).to(device)

    # Ensure dummy labels are within range
    dummy_labels = torch.clamp(dummy_labels, 0, num_classes - 1)

    output_train = model(dummy_input, dummy_labels)
    assert output_train.shape == (
        4,
        num_classes,
    ), f"Output shape mismatch: {output_train.shape}"

    # Assertion: Check Forward Pass (Inference Mode)
    output_infer = model(dummy_input)
    assert output_infer.shape == (4, num_classes), "Inference output shape mismatch"

    # Assertion: Check Feature Extraction
    features = model.extract_features(dummy_input)
    assert features.shape == (
        4,
        Config.EMBEDDING_DIM,
    ), f"Feature shape mismatch: {features.shape}"

    # Verify normalization (L2 norm should be approx 1.0)
    norms = torch.norm(features, p=2, dim=1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), "Features are not normalized"

    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 4. Metric Verification (MAP@K)
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Metric Calculation...")

    # Case 1: Perfect prediction
    # Target: [0, 1]
    # Preds: [[0, 2, 3, 4, 5], [1, 0, 2, 3, 4]]
    t_targets = torch.tensor([0, 1])
    t_preds = torch.tensor([[0, 2, 3, 4, 5], [1, 0, 2, 3, 4]])
    score = mapk(t_targets, t_preds, k=5)
    assert abs(score - 1.0) < 1e-6, f"MAP@5 should be 1.0, got {score}"

    # Case 2: No match
    t_preds_wrong = torch.tensor([[9, 2, 3, 4, 5], [9, 0, 2, 3, 4]])
    score_wrong = mapk(t_targets, t_preds_wrong, k=5)
    assert score_wrong == 0.0, f"MAP@5 should be 0.0, got {score_wrong}"

    print("MAP@K function verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Loop (1 Epoch on Subset)...")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=1
    )

    assert not np.isnan(train_loss), "Training loss is NaN"
    print(f"Train Loop Finished. Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    # -------------------------------------------------------------------------
    # 6. Validation Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Running Validation...")

    val_map = validate(model, val_loader, device, criterion)

    assert 0.0 <= val_map <= 1.0, f"Validation MAP@5 out of range: {val_map}"
    print(f"Validation Finished. MAP@5: {val_map:.5f}")

    # Save a dummy checkpoint to simulate the workflow
    torch.save(
        {"state_dict": model.state_dict(), "best_map5": val_map}, Config.BEST_MODEL_PATH
    )
    assert os.path.exists(Config.BEST_MODEL_PATH), "Checkpoint file not created."

    # -------------------------------------------------------------------------
    # 7. Inference & Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference & Submission...")

    predict_and_submit(
        model, test_loader, encoder_classes, device, Config.SUBMISSION_PATH
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check format
    assert list(sub_df.columns) == ["image", "hotel_id"], "Submission columns mismatch"
    assert (
        len(sub_df) == subset_size
    ), f"Expected {subset_size} predictions, got {len(sub_df)}"

    # Check content format (space delimited IDs)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction is not a string"
    assert (
        len(sample_pred.split(" ")) == 5
    ), f"Prediction does not contain 5 IDs: {sample_pred}"

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
