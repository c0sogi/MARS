import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    competition_loss,
    get_competition_weights,
)
from library.models import FractureClassifier
from library.dataset import process_slice_metadata, FractureCropDataset
from library.segmentation_engine import generate_spine_coordinates

DEVICE = Config.DEVICE


def train_classifier(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Trains the Stage 2 Fracture Classifier.
    """
    print(f"Starting Classifier Training on {DEVICE}...")
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_meta = train_meta.head(Config.DEBUG_DATASET_SIZE)
        val_meta = val_meta.head(Config.DEBUG_DATASET_SIZE)

    # 2. Process Slices
    # Load bounding boxes for training labels
    bbox_df = None
    if os.path.exists(Config.TRAIN_BBOX_PATH):
        bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

    train_slice_df = process_slice_metadata(
        train_meta, bbox_df, mode="train", load_cached_data=load_cached_data
    )
    val_slice_df = process_slice_metadata(
        val_meta, bbox_df, mode="val", load_cached_data=load_cached_data
    )

    # 3. Generate/Load Spine Coordinates
    # We need coordinates for both train and val to crop correctly
    train_coords = generate_spine_coordinates(
        train_meta, mode="train", load_cached_data=load_cached_data
    )
    val_coords = generate_spine_coordinates(
        val_meta, mode="val", load_cached_data=load_cached_data
    )

    # 4. Datasets & Loaders
    # Train dataset uses balanced sampling
    train_dataset = FractureCropDataset(
        train_slice_df, coords_map=train_coords, mode="train"
    )
    # Val dataset uses sequential sampling (no balancing)
    val_dataset = FractureCropDataset(val_slice_df, coords_map=val_coords, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 5. Model Setup
    model = FractureClassifier(pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Slice-level loss: BCE
    criterion = nn.BCEWithLogitsLoss()

    best_metric = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_classifier.pth")

    # 6. Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation Step
        model.eval()

        # We need to aggregate predictions by StudyInstanceUID to calculate the competition metric
        val_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                logits = model(images)
                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())

        if len(val_probs) > 0:
            val_probs = np.concatenate(val_probs, axis=0)
        else:
            val_probs = np.zeros((0, Config.NUM_CLASSES))

        # Assign to dataframe
        # Create a copy to avoid SettingWithCopy warnings on the cached df
        temp_val_df = val_slice_df.copy()
        pred_cols = [f"pred_{c}" for c in Config.TARGET_COLS]

        # Ensure lengths match
        if len(temp_val_df) == len(val_probs):
            temp_val_df[pred_cols] = val_probs

            # Aggregate: Max Pooling per Study
            # Group by StudyInstanceUID and take max of prediction columns
            study_preds = (
                temp_val_df.groupby("StudyInstanceUID")[pred_cols].max().reset_index()
            )

            # Merge with Ground Truth (val_meta)
            # val_meta has columns: StudyInstanceUID, C1, ..., patient_overall
            val_merged = pd.merge(
                val_meta, study_preds, on="StudyInstanceUID", how="inner"
            )

            # Calculate Competition Loss
            # Extract Preds and Targets
            y_pred_tensor = torch.tensor(
                val_merged[pred_cols].values, dtype=torch.float32, device=DEVICE
            )
            y_true_tensor = torch.tensor(
                val_merged[Config.TARGET_COLS].values,
                dtype=torch.float32,
                device=DEVICE,
            )

            val_metric = competition_loss(y_pred_tensor, y_true_tensor).item()
        else:
            print("Warning: Mismatch between validation set size and predictions.")
            val_metric = float("inf")

        print(
            f"Epoch {epoch}/{num_epochs} - Train Loss (BCE): {train_loss:.6f} - Val Metric (Weighted LogLoss): {val_metric:.6f}"
        )

        # Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            save_checkpoint(model, optimizer, epoch, best_metric, checkpoint_path)
            print(f"  Saved new best model with metric: {best_metric:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Classifier training complete.")


def inference_and_submission(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Generates predictions for the test set and creates the submission file.
    """
    print("Starting Inference and Submission Generation...")

    # 1. Load Test Metadata
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Process Test Slices
    test_slice_df = process_slice_metadata(
        test_meta, bbox_df=None, mode="test", load_cached_data=load_cached_data
    )

    # 3. Generate Test Coordinates
    test_coords = generate_spine_coordinates(
        test_meta, mode="test", load_cached_data=load_cached_data
    )

    # 4. Dataset & Loader
    test_dataset = FractureCropDataset(
        test_slice_df, coords_map=test_coords, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Load Model
    model = FractureClassifier(pretrained=False).to(DEVICE)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_classifier.pth")

    if os.path.exists(checkpoint_path):
        model, _, _, _ = load_checkpoint(model, None, checkpoint_path, device=DEVICE)
        print("Loaded trained classifier weights.")
    else:
        print("Warning: No trained classifier found. Using random weights.")

    model.eval()

    # 6. Inference Loop
    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        all_probs = np.zeros((0, Config.NUM_CLASSES))

    # 7. Aggregation
    temp_test_df = test_slice_df.copy()
    pred_cols = Config.TARGET_COLS  # ["C1", ..., "patient_overall"]

    if len(temp_test_df) == len(all_probs):
        temp_test_df[pred_cols] = all_probs

        # Max Pooling per Study
        study_preds = (
            temp_test_df.groupby("StudyInstanceUID")[pred_cols].max().reset_index()
        )

        # 8. Format Submission
        # We need to transform from wide format (columns) to long format (rows)
        # Row ID format: [StudyUID]_[TargetName]

        submission_rows = []

        for _, row in study_preds.iterrows():
            uid = row["StudyInstanceUID"]
            for col in Config.TARGET_COLS:
                row_id = f"{uid}_{col}"
                prob = row[col]
                submission_rows.append({"row_id": row_id, "fractured": prob})

        submission_df = pd.DataFrame(submission_rows)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(
            f"Submission saved to {Config.SUBMISSION_PATH} with {len(submission_df)} rows."
        )
    else:
        print("Error: Mismatch between test slices and predictions.")
