import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import load_data, SIIMDataset, get_transforms
from library.model import StochasticResNet34UNet
from library.utils import seed_everything, AverageMeter, mask2bbox, calculate_map


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Loss functions
    criterion_study = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        with autocast():
            study_logits, mask_logits = model(images)

            # Study Loss (Multi-class classification)
            # labels are one-hot, CrossEntropyLoss expects class indices
            study_targets = torch.argmax(labels, dim=1)
            loss_study = criterion_study(study_logits, study_targets)

            # Mask Loss
            loss_mask = criterion_mask(mask_logits, masks)

            # Weighted Sum
            loss = (Config.STUDY_LOSS_WEIGHT * loss_study) + (
                Config.IMAGE_LOSS_WEIGHT * loss_mask
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns loss, image mAP, and study accuracy.
    """
    model.eval()
    losses = AverageMeter()

    criterion_study = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    # Store predictions for mAP calculation
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []

    study_preds = []
    study_gts = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            orig_dims = batch["orig_dim"].numpy()  # (N, 2) -> h, w

            study_logits, mask_logits = model(images)

            # Loss
            study_targets = torch.argmax(labels, dim=1)
            loss_study = criterion_study(study_logits, study_targets)
            loss_mask = criterion_mask(mask_logits, masks)
            loss = (Config.STUDY_LOSS_WEIGHT * loss_study) + (
                Config.IMAGE_LOSS_WEIGHT * loss_mask
            )
            losses.update(loss.item(), images.size(0))

            # Process Study Predictions
            probs = torch.softmax(study_logits, dim=1)
            study_preds.append(probs.cpu().numpy())
            study_gts.append(labels.cpu().numpy())

            # Process Mask Predictions for mAP
            mask_probs = torch.sigmoid(mask_logits).cpu().numpy()  # (N, 1, H, W)
            gt_masks = masks.cpu().numpy()

            for i in range(images.size(0)):
                # Resize mask prob back to original size for accurate box extraction
                # Validation mAP is calculated on 512x512 here for efficiency during training
                # but using original dimensions is also possible. Keeping consistent with provided logic.
                curr_mask = mask_probs[i, 0]

                boxes = mask2bbox(curr_mask, threshold=0.5)

                # Normalize scores for boxes (using mean pixel value in box)
                scores = []
                for b in boxes:
                    x1, y1, x2, y2 = b
                    # Clip coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(Config.IMG_SIZE, x2), min(Config.IMG_SIZE, y2)
                    if x2 > x1 and y2 > y1:
                        score = np.mean(curr_mask[y1:y2, x1:x2])
                        scores.append(score)
                    else:
                        scores.append(0.0)

                if len(boxes) > 0:
                    pred_boxes_list.append(np.array(boxes))
                    pred_scores_list.append(np.array(scores))
                    pred_labels_list.append(np.zeros(len(boxes)))  # Class 0 for opacity
                else:
                    pred_boxes_list.append(np.empty((0, 4)))
                    pred_scores_list.append(np.array([]))
                    pred_labels_list.append(np.array([]))

                # GT Boxes
                curr_gt_mask = gt_masks[i, 0]
                g_boxes = mask2bbox(curr_gt_mask, threshold=0.5)
                if len(g_boxes) > 0:
                    gt_boxes_list.append(np.array(g_boxes))
                    gt_labels_list.append(np.zeros(len(g_boxes)))
                else:
                    gt_boxes_list.append(np.empty((0, 4)))
                    gt_labels_list.append(np.array([]))

    # Calculate mAP
    image_map = calculate_map(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        num_classes=1,
    )

    # Study Accuracy
    if len(study_preds) > 0:
        study_preds = np.concatenate(study_preds)
        study_gts = np.concatenate(study_gts)
        study_acc = (
            np.argmax(study_preds, axis=1) == np.argmax(study_gts, axis=1)
        ).mean()
    else:
        study_acc = 0.0

    return losses.avg, image_map, study_acc


def train_model():
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Data Loading
    train_data = load_data(Config.TRAIN_METADATA, "train", load_cached_data=True)
    val_data = load_data(Config.VAL_METADATA, "val", load_cached_data=True)

    train_dataset = SIIMDataset(train_data, "train", transform=get_transforms("train"))
    val_dataset = SIIMDataset(val_data, "val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = StochasticResNet34UNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Total steps for Cosine Annealing
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS * steps_per_epoch, eta_min=Config.ETA_MIN
    )

    scaler = GradScaler()

    best_score = 0.0
    patience = 7
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        val_loss, val_map, val_acc = evaluate(model, val_loader, device)

        # Composite score: Average of Image mAP and Study Acc
        composite_score = (val_map + val_acc) / 2

        print(
            f"Epoch {epoch+1} Train Loss: {train_loss} Val Loss: {val_loss} "
            f"Image mAP: {val_map} Study Acc: {val_acc} Score: {composite_score}"
        )

        if composite_score > best_score:
            best_score = composite_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! Score: {best_score}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training Complete. Best Score: {best_score}")


def predict():
    """
    Runs inference on the test set and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Data
    test_data = load_data(Config.TEST_METADATA, "test", load_cached_data=True)
    test_dataset = SIIMDataset(test_data, "test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = StochasticResNet34UNet().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using random weights for inference.")

    model.eval()

    study_classes = ["negative", "typical", "indeterminate", "atypical"]

    # Load test metadata to map indices to IDs
    test_df = pd.read_csv(Config.TEST_METADATA)

    final_results = []
    ptr = 0

    print("Running Inference...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            orig_dims = batch["orig_dim"].numpy()
            batch_size = images.size(0)

            # TTA: Original + Horizontal Flip
            # Forward Original
            study_logits, mask_logits = model(images)
            study_probs = torch.softmax(study_logits, dim=1)
            mask_probs = torch.sigmoid(mask_logits)

            # Forward Flip
            images_flip = torch.flip(images, dims=[3])
            study_logits_f, mask_logits_f = model(images_flip)
            study_probs_f = torch.softmax(study_logits_f, dim=1)
            mask_probs_f = torch.sigmoid(mask_logits_f)
            mask_probs_f = torch.flip(mask_probs_f, dims=[3])

            # Average
            avg_study_probs = (study_probs + study_probs_f) / 2.0
            avg_mask_probs = (mask_probs + mask_probs_f) / 2.0

            avg_study_probs = avg_study_probs.cpu().numpy()
            avg_mask_probs = avg_mask_probs.cpu().numpy()

            # Process batch
            for i in range(batch_size):
                # Get metadata from DF
                if ptr < len(test_df):
                    row = test_df.iloc[ptr]
                    ptr += 1
                    study_id = row["study_id"]
                    image_id = row["image_id"]
                else:
                    break  # Should not happen if loader and df are aligned

                h, w = orig_dims[i]

                # 1. Study Prediction
                # Format: "class conf 0 0 1 1" for all classes
                study_pred_strs = []
                for idx, cls_name in enumerate(study_classes):
                    conf = avg_study_probs[i, idx]
                    study_pred_strs.append(f"{cls_name} {conf:.6f} 0 0 1 1")
                study_str = " ".join(study_pred_strs)

                final_results.append(
                    {"id": f"{study_id}_study", "PredictionString": study_str}
                )

                # 2. Image Prediction
                # Logic: If 'negative' is the highest class, predict none.
                neg_idx = 0  # 'negative' is first in our list
                pred_class_idx = np.argmax(avg_study_probs[i])

                if pred_class_idx == neg_idx:
                    img_str = "none 1 0 0 1 1"
                else:
                    mask = avg_mask_probs[i, 0]
                    # Resize mask to original image dimensions for correct box coordinates
                    mask_resized = cv2.resize(
                        mask, (w, h), interpolation=cv2.INTER_LINEAR
                    )

                    boxes = mask2bbox(mask_resized, threshold=0.5)

                    if len(boxes) == 0:
                        img_str = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            # Calculate score for this box
                            # Use mean of probability map within box
                            box_score = np.mean(mask_resized[y1:y2, x1:x2])
                            box_strs.append(
                                f"opacity {box_score:.6f} {x1} {y1} {x2} {y2}"
                            )
                        img_str = " ".join(box_strs)

                final_results.append(
                    {"id": f"{image_id}_image", "PredictionString": img_str}
                )

    submission_df = pd.DataFrame(final_results)
    # Remove duplicates if any
    submission_df = submission_df.drop_duplicates(subset=["id"])

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
