import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

from library.config import Config
from library.dataset import ChestXRayDataset
from library.utils import (
    collate_fn,
    format_image_prediction_string,
    format_study_prediction_string,
)


def get_one_stage_detector():
    """
    Constructs the One-Stage Detector (RetinaNet) adapted for the specific classes.
    Replaces the head of a pre-trained RetinaNet to match the 4 classes (0=BG, 1=Typical, 2=Indeterminate, 3=Atypical).
    """
    # Load model with default COCO weights
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights)

    # Replace the classification head
    # Num classes = 3 specific findings + 1 background = 4
    num_classes = Config.NUM_CLASSES + 1

    # Get input features from the backbone
    in_channels = model.backbone.out_channels

    # Get the number of anchors from the existing head
    num_anchors = model.head.classification_head.num_anchors

    # Create new head
    model.head = RetinaNetHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )

    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for i, (images, targets, _) in enumerate(dataloader):
        images = images.to(device)
        # Move targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def validate_one_epoch(model, dataloader, device):
    """
    Runs validation.
    Note: Torchvision detection models do not return loss in eval() mode.
    We run in train() mode with no_grad() to compute validation loss.
    """
    model.train()  # Keep in train mode to get loss dict
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def train_model(load_cached_data=True):
    """
    Main training pipeline with Early Stopping.
    """
    print(f"Starting training on device: {Config.DEVICE}")

    # 1. Prepare Data
    train_dataset = ChestXRayDataset(split="train", load_cached_data=load_cached_data)
    val_dataset = ChestXRayDataset(split="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Prepare Model & Optimizer
    model = get_one_stage_detector()
    model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, Config.DEVICE, epoch
        )
        val_loss = validate_one_epoch(model, val_loader, Config.DEVICE)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! (Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")


def predict_and_submit(load_cached_data=True):
    """
    Runs inference on the test set and generates the submission file.
    """
    print("Starting inference...")

    # 1. Load Data
    test_dataset = ChestXRayDataset(split="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Load Model
    model = get_one_stage_detector()
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "Warning: No trained model found. Using random weights (for debugging only)."
        )

    model.to(Config.DEVICE)
    model.eval()

    results = []

    # 3. Inference Loop
    with torch.no_grad():
        for images, _, image_ids in test_loader:
            images = images.to(Config.DEVICE)

            # RetinaNet returns a list of dicts: [{'boxes': ..., 'scores': ..., 'labels': ...}, ...]
            detections = model(images)

            # Move to CPU for processing
            detections = [
                {k: v.cpu().numpy() for k, v in d.items()} for d in detections
            ]

            for img_id, det in zip(image_ids, detections):
                boxes = det["boxes"]
                scores = det["scores"]
                labels = det["labels"]

                # Filter by confidence threshold
                mask = scores > Config.CONF_THRESHOLD
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                # Study-level prediction string
                # Note: study ID is usually img_id.replace("_image", "_study") or handled via mapping.
                # The sample submission expects rows for both study and image.
                # Based on dataset.py, image_id is the base filename without extension.
                # We need to reconstruct the IDs required by submission.csv.
                # The sample submission has format: 'id', 'PredictionString'
                # IDs look like '2b95d54e4be65_study' and '2b95d54e4be68_image'.
                # Our test dataset metadata provides 'study_id' and 'image_id'.
                # However, ChestXRayDataset returns just 'image_id'.
                # We will generate rows for the image_id provided.
                # We need to look up the study_id for this image_id to generate the study row.
                # To be robust, we'll generate the image prediction here.
                # The study prediction requires aggregation if multiple images per study,
                # but usually test set is 1 image per study or we treat them independently.
                # Given the structure, we will generate entries for:
                # 1. {image_id}_image
                # 2. {study_id}_study (We need to fetch study_id from metadata)

                # Format strings
                image_pred_str = format_image_prediction_string(boxes, scores)
                study_pred_str = format_study_prediction_string(boxes, scores, labels)

                results.append(
                    {
                        "id": f"{img_id}_image",
                        "PredictionString": image_pred_str,
                        "study_pred": study_pred_str,
                        "image_id_raw": img_id,
                    }
                )

    # 4. Post-processing for Submission
    # We need to merge with metadata to get study_ids
    df_results = pd.DataFrame(results)

    # Load test metadata to map image_id to study_id
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Merge to get study_id
    df_merged = pd.merge(
        df_results,
        df_test_meta[["image_id", "study_id"]],
        left_on="image_id_raw",
        right_on="image_id",
        how="left",
    )

    submission_rows = []

    # Add Image Rows
    for _, row in df_merged.iterrows():
        submission_rows.append(
            {"id": row["id"], "PredictionString": row["PredictionString"]}
        )

    # Add Study Rows
    # Group by study_id and take the "most confident" prediction if multiple images exist
    # (Or just take the first one if 1:1 mapping)
    # The metric requires at least one label. Our format_study_prediction_string handles this.
    study_groups = df_merged.groupby("study_id")

    for study_id, group in study_groups:
        # If multiple images, we could merge boxes, but here we pick the one with highest confidence finding
        # Parsing the study_pred string is messy, so we'll rely on the fact that
        # format_study_prediction_string puts the highest confidence class first.
        # We'll just take the first image's prediction for simplicity, or implement a simple max logic.
        # Given the task constraints, taking the prediction from the image with the most boxes
        # or highest confidence score is a valid heuristic.

        # Simple heuristic: Take the row that is NOT "negative" if possible
        candidates = group["study_pred"].tolist()
        selected_pred = candidates[0]

        for pred in candidates:
            if "negative" not in pred:
                selected_pred = pred
                break

        submission_rows.append(
            {"id": f"{study_id}_study", "PredictionString": selected_pred}
        )

    # 5. Save Submission
    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(
        f"Submission saved to {Config.SUBMISSION_FILE_PATH} with {len(df_submission)} rows."
    )
