import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2
import time

from library.config import Config
from library.dataset import SIIMDataset
from library.model import ResNet18UNet
from library.utils import seed_everything, prepare_gt_from_metadata, calculate_map

# Mapping from class index to submission string
CLASS_ID_TO_NAME = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}


def get_class_name(idx):
    return CLASS_ID_TO_NAME.get(idx, "negative")


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0

    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.BCEWithLogitsLoss()

    dataset_size = len(loader.dataset)
    num_batches = len(loader)

    for batch_idx, data in enumerate(loader):
        images = data["image"].to(device)
        labels = data["labels"].to(device)  # One-hot or multi-label float
        masks = data["mask"].to(device)

        # Convert labels to class indices for CrossEntropy
        # labels in dataset are [Neg, Typ, Ind, Atyp]. Argmax gives the index.
        cls_targets = torch.argmax(labels, dim=1)

        optimizer.zero_grad()

        cls_logits, seg_logits = model(images)

        # Calculate losses
        cls_loss = cls_criterion(cls_logits, cls_targets)
        seg_loss = seg_criterion(seg_logits, masks)

        # Weighted sum
        total_loss = (cls_loss * Config.CLS_LOSS_WEIGHT) + (
            seg_loss * Config.SEG_LOSS_WEIGHT
        )

        total_loss.backward()
        optimizer.step()

        # Update scheduler if it steps per batch (CosineAnnealing usually per epoch, but check usage)
        # We will step scheduler per epoch in run(), so skip here.

        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_seg_loss += seg_loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    epoch_cls_loss = running_cls_loss / dataset_size
    epoch_seg_loss = running_seg_loss / dataset_size

    print(
        f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Seg: {epoch_seg_loss:.4f})"
    )

    return epoch_loss


def evaluate(model, loader, device, metadata_df):
    model.eval()

    predictions = []

    print("Evaluating on validation set...")

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            image_ids = data["image_id"]
            study_ids = data["study_id"]

            # Test-Time Augmentation (Horizontal Flip)
            # Cite solution_lesson_node_00023
            images_flip = torch.flip(images, dims=[3])

            # Forward pass original
            cls_logits, seg_logits = model(images)
            # Forward pass flipped
            cls_logits_f, seg_logits_f = model(images_flip)

            # Average Predictions
            cls_probs = (
                torch.softmax(cls_logits, dim=1) + torch.softmax(cls_logits_f, dim=1)
            ) / 2.0

            seg_probs_orig = torch.sigmoid(seg_logits)
            seg_probs_flip = torch.sigmoid(seg_logits_f)
            seg_probs_flip_back = torch.flip(seg_probs_flip, dims=[3])
            seg_probs = (seg_probs_orig + seg_probs_flip_back) / 2.0

            # Study Predictions
            pred_cls_ids = torch.argmax(cls_probs, dim=1).cpu().numpy()
            pred_cls_confs = torch.max(cls_probs, dim=1).values.cpu().numpy()

            # Segmentation Predictions
            seg_masks = (seg_probs > 0.5).float().cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):
                img_id = image_ids[i]
                std_id = study_ids[i]

                # Study Level Prediction String
                # Format: "class conf 0 0 1 1"
                # We need to output predictions for the study.
                # The metric evaluates study and image separately.
                # For validation mAP calculation, we need to format predictions correctly.

                # 1. Study Prediction
                pred_label_name = get_class_name(pred_cls_ids[i])
                study_pred_str = f"{pred_label_name} {pred_cls_confs[i]:.6f} 0 0 1 1"
                predictions.append(
                    {"Id": f"{std_id}_study", "PredictionString": study_pred_str}
                )

                # 2. Image Prediction
                # Gating: If Negative, force none
                if pred_cls_ids[i] == 0:  # 0 is Negative for Pneumonia
                    image_pred_str = "none 1.0 0 0 1 1"
                else:
                    # Extract boxes
                    mask = seg_masks[i, 0]  # (H, W)
                    prob_map = seg_probs_np[i, 0]

                    # Resize mask to original image size?
                    # The model outputs 512x512. The GT is in original coordinates.
                    # We need to scale boxes back to original size.
                    # However, calculate_map works with relative or absolute?
                    # The dataset returns 512x512 images. The GT boxes in metadata are absolute.
                    # We need original dimensions to scale back.
                    # Metadata DF has this info if we look up by image_id, or we can just rely on IoU
                    # if we scale GT to 512x512.
                    # Easier: Scale Predicted Boxes to Original Size.
                    # We need original dims. The loader doesn't return them by default.
                    # Let's modify logic: The mAP calculation compares boxes.
                    # If we use the GT dataframe prepared by `prepare_gt_from_metadata`, it uses the raw 'label' string.
                    # The raw 'label' string has boxes in ORIGINAL coordinates.
                    # So we MUST scale our predictions to ORIGINAL coordinates.

                    # We can get original dims from metadata_df
                    row = metadata_df[metadata_df["image_id"] == img_id].iloc[0]
                    # We don't have width/height columns in provided metadata csv directly visible in description,
                    # but we can infer or we might have to read the file.
                    # Wait, the analysis script showed width/height are not in metadata CSV columns explicitly,
                    # but `read_xray` reads them.
                    # To avoid reading every DICOM again, we can assume we need to handle this.
                    # Actually, `prepare_gt_from_metadata` uses the `label` column which is already correct.
                    # We need to know the scale factor.
                    # Since we don't have width/height in metadata, let's look at the `boxes` column in metadata.
                    # It has x, y, w, h. We can't easily get image size if there are no boxes.
                    # CRITICAL: We need original image dimensions to scale back.
                    # The SIIMDataset `__getitem__` reads the image.
                    # Let's assume for validation we might need to read the image or cache dims.
                    # For efficiency, we will assume 1.0 scale (512x512) for calculation IF we scaled GT.
                    # But we didn't scale GT.
                    # Solution: In `evaluate`, we can't easily get original dims without reading files.
                    # However, we can use the fact that `SIIMDataset` loads the image.
                    # We can modify `SIIMDataset` to return `orig_size`? No, I cannot modify library files.

                    # Workaround: Read the image file header again using pydicom to get rows/cols.
                    # It's slow but necessary for correct mAP.
                    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
                    dcm_header = cv2.imread(file_path)  # No, it's dicom.
                    # Use pydicom but only header
                    import pydicom

                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    orig_h, orig_w = ds.Rows, ds.Columns

                    scale_x = orig_w / Config.IMG_SIZE
                    scale_y = orig_h / Config.IMG_SIZE

                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    box_preds = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Filter small noise
                        if w < 2 or h < 2:
                            continue

                        # Calculate confidence: mean probability in the box
                        # Slice from prob_map
                        roi = prob_map[y : y + h, x : x + w]
                        conf = np.mean(roi)

                        # Scale to original
                        x_orig = x * scale_x
                        y_orig = y * scale_y
                        w_orig = w * scale_x
                        h_orig = h * scale_y

                        # Format: opacity conf xmin ymin xmax ymax
                        # xmax = x + w
                        box_preds.append(
                            f"opacity {conf:.4f} {x_orig:.1f} {y_orig:.1f} {x_orig+w_orig:.1f} {y_orig+h_orig:.1f}"
                        )

                    if not box_preds:
                        image_pred_str = "none 1.0 0 0 1 1"
                    else:
                        image_pred_str = " ".join(box_preds)

                predictions.append(
                    {"Id": f"{img_id}_image", "PredictionString": image_pred_str}
                )

    pred_df = pd.DataFrame(predictions)

    # Prepare GT
    gt_df = prepare_gt_from_metadata(
        metadata_df, load_cached_data=True, cache_dir=Config.WORKING_DIR
    )

    # Calculate mAP
    # We need to filter gt_df to only include IDs in pred_df (validation set)
    val_ids = set(pred_df["Id"].unique())
    gt_df_val = gt_df[gt_df["Id"].isin(val_ids)].copy()

    map_score = calculate_map(pred_df, gt_df_val, iou_threshold=0.5)
    print(f"Validation mAP: {map_score:.6f}")

    return map_score


def predict_test(model, loader, device, metadata_df):
    model.eval()
    predictions = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            image_ids = data["image_id"]
            study_ids = data["study_id"]

            # Test-Time Augmentation (Horizontal Flip)
            # Cite solution_lesson_node_00023
            images_flip = torch.flip(images, dims=[3])

            # Forward pass original
            cls_logits, seg_logits = model(images)
            # Forward pass flipped
            cls_logits_f, seg_logits_f = model(images_flip)

            # Average Predictions
            cls_probs = (
                torch.softmax(cls_logits, dim=1) + torch.softmax(cls_logits_f, dim=1)
            ) / 2.0

            seg_probs_orig = torch.sigmoid(seg_logits)
            seg_probs_flip = torch.sigmoid(seg_logits_f)
            seg_probs_flip_back = torch.flip(seg_probs_flip, dims=[3])
            seg_probs = (seg_probs_orig + seg_probs_flip_back) / 2.0

            # Study
            pred_cls_ids = torch.argmax(cls_probs, dim=1).cpu().numpy()
            pred_cls_confs = torch.max(cls_probs, dim=1).values.cpu().numpy()

            # Seg
            seg_masks = (seg_probs > 0.5).float().cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):
                img_id = image_ids[i]
                std_id = study_ids[i]

                # Study String
                pred_label_name = get_class_name(pred_cls_ids[i])
                study_pred_str = f"{pred_label_name} {pred_cls_confs[i]:.6f} 0 0 1 1"
                predictions.append(
                    {"Id": f"{std_id}_study", "PredictionString": study_pred_str}
                )

                # Image String
                if pred_cls_ids[i] == 0:
                    image_pred_str = "none 1.0 0 0 1 1"
                else:
                    mask = seg_masks[i, 0]
                    prob_map = seg_probs_np[i, 0]

                    # Get original dims
                    row = metadata_df[metadata_df["image_id"] == img_id].iloc[0]
                    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
                    import pydicom

                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    orig_h, orig_w = ds.Rows, ds.Columns

                    scale_x = orig_w / Config.IMG_SIZE
                    scale_y = orig_h / Config.IMG_SIZE

                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    box_preds = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w < 2 or h < 2:
                            continue

                        roi = prob_map[y : y + h, x : x + w]
                        conf = np.mean(roi)

                        x_orig = x * scale_x
                        y_orig = y * scale_y
                        w_orig = w * scale_x
                        h_orig = h * scale_y

                        box_preds.append(
                            f"opacity {conf:.4f} {x_orig:.1f} {y_orig:.1f} {x_orig+w_orig:.1f} {y_orig+h_orig:.1f}"
                        )

                    if not box_preds:
                        image_pred_str = "none 1.0 0 0 1 1"
                    else:
                        image_pred_str = " ".join(box_preds)

                predictions.append(
                    {"Id": f"{img_id}_image", "PredictionString": image_pred_str}
                )

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    seed_everything(Config.SEED)

    # 1. Data Loaders
    train_dataset = SIIMDataset("train", load_cached_data=True)
    val_dataset = SIIMDataset("val", load_cached_data=True)

    # Debug mode
    if Config.DEBUG:
        train_dataset.df = train_dataset.df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_dataset.df = val_dataset.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model & Optimizer
    device = torch.device(Config.DEVICE)
    model = ResNet18UNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 3. Training Loop
    best_map = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Step scheduler
        scheduler.step()

        # Validate
        val_map = evaluate(model, val_loader, device, val_dataset.df)

        # Checkpointing
        if val_map > best_map:
            print(f"mAP improved from {best_map:.6f} to {val_map:.6f}. Saving model...")
            best_map = val_map
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"mAP did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 4. Test Prediction
    print("Training complete. Loading best model for testing...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model.")

    test_dataset = SIIMDataset("test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    predict_test(model, test_loader, device, test_dataset.df)

    print("Done.")
