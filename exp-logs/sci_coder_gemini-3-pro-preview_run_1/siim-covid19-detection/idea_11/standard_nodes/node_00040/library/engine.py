import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from library.dataset import get_dataset
from library.model import ResNet18UNet
from library.utils import seed_everything, get_box_from_mask, map_calculation

# Constants
CLASS_MAP = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}

OUTPUT_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    for batch_idx, (images, masks, labels) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            mask_preds, cls_logits = model(images)

            # Study Loss: labels are one-hot (N, 4), convert to indices (N,)
            target_cls = torch.argmax(labels, dim=1)
            loss_cls = criterion_cls(cls_logits, target_cls)

            # Segmentation Loss
            loss_seg = criterion_seg(mask_preds, masks)

            # Composite Loss (1:10 ratio)
            loss = loss_cls + 10.0 * loss_seg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device, df_val):
    model.eval()
    running_loss = 0.0

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    pred_rows = []

    # We rely on the loader being shuffle=False and aligned with df_val

    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            mask_preds, cls_logits = model(images)

            # Loss Calculation
            target_cls = torch.argmax(labels, dim=1)
            loss_cls = criterion_cls(cls_logits, target_cls)
            loss_seg = criterion_seg(mask_preds, masks)
            loss = loss_cls + 10.0 * loss_seg
            running_loss += loss.item()

            # Predictions for mAP
            probs = torch.softmax(cls_logits, dim=1)
            conf_cls, pred_cls_idx = torch.max(probs, dim=1)
            mask_probs = torch.sigmoid(mask_preds)

            bs = images.size(0)
            start_idx = batch_idx * loader.batch_size

            for i in range(bs):
                global_idx = start_idx + i
                # Safety check for index
                if global_idx >= len(df_val):
                    break

                row = df_val.iloc[global_idx]
                study_id = row["study_id"]
                image_id = row["image_id"]

                p_cls = int(pred_cls_idx[i].item())
                p_conf = conf_cls[i].item()
                p_mask = mask_probs[i, 0].cpu().numpy()

                # Study Prediction
                study_pred_str = f"{CLASS_MAP[p_cls]} {p_conf:.6f} 0 0 1 1"
                pred_rows.append(
                    {"id": f"{study_id}_study", "PredictionString": study_pred_str}
                )

                # Image Prediction (Gated)
                if p_cls == 0:  # Negative
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    boxes = get_box_from_mask(p_mask, threshold=0.5)
                    if len(boxes) == 0:
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            # Format: opacity conf x1 y1 x2 y2
                            box_strs.append(
                                f"opacity {p_conf:.6f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}"
                            )
                        image_pred_str = " ".join(box_strs)

                pred_rows.append(
                    {"id": f"{image_id}_image", "PredictionString": image_pred_str}
                )

    pred_df = pd.DataFrame(pred_rows)

    # Construct Ground Truth DataFrame for mAP calculation
    gt_rows = []
    for _, row in df_val.iterrows():
        # Study GT
        true_cls_idx = 0
        if row["Negative for Pneumonia"] == 1:
            true_cls_idx = 0
        elif row["Typical Appearance"] == 1:
            true_cls_idx = 1
        elif row["Indeterminate Appearance"] == 1:
            true_cls_idx = 2
        elif row["Atypical Appearance"] == 1:
            true_cls_idx = 3

        gt_rows.append(
            {
                "id": f"{row['study_id']}_study",
                "PredictionString": f"{CLASS_MAP[true_cls_idx]} 1 0 0 1 1",
            }
        )

        # Image GT
        gt_rows.append(
            {"id": f"{row['image_id']}_image", "PredictionString": row["label"]}
        )

    gt_df = pd.DataFrame(gt_rows)

    # Calculate mAP
    metric = map_calculation(gt_df, pred_df, iou_threshold=0.5)

    return running_loss / len(loader), metric


def predict_test(model, loader, device, df_test):
    model.eval()
    pred_rows = []

    # Cite solution_lesson_node_00023: Test-Time Augmentation (TTA)
    with torch.no_grad():
        for batch_idx, (images, ids) in enumerate(loader):
            images = images.to(device)

            # 1. Forward Pass (Original)
            mask_preds1, cls_logits1 = model(images)

            # 2. Forward Pass (Horizontal Flip)
            images_flip = torch.flip(images, dims=[3])
            mask_preds2, cls_logits2 = model(images_flip)

            # 3. Average Predictions
            probs1 = torch.softmax(cls_logits1, dim=1)
            probs2 = torch.softmax(cls_logits2, dim=1)
            probs = (probs1 + probs2) / 2.0

            mask_probs1 = torch.sigmoid(mask_preds1)
            mask_probs2 = torch.sigmoid(mask_preds2)
            # Flip back the mask
            mask_probs2 = torch.flip(mask_probs2, dims=[3])
            mask_probs = (mask_probs1 + mask_probs2) / 2.0

            conf_cls, pred_cls_idx = torch.max(probs, dim=1)

            bs = images.size(0)
            start_idx = batch_idx * loader.batch_size

            for i in range(bs):
                global_idx = start_idx + i
                if global_idx >= len(df_test):
                    break

                row = df_test.iloc[global_idx]
                study_id = row["study_id"]
                image_id = row["image_id"]

                p_cls = int(pred_cls_idx[i].item())
                p_conf = conf_cls[i].item()
                p_mask = mask_probs[i, 0].cpu().numpy()

                # Study
                pred_rows.append(
                    {
                        "id": f"{study_id}_study",
                        "PredictionString": f"{CLASS_MAP[p_cls]} {p_conf:.6f} 0 0 1 1",
                    }
                )

                # Image
                if p_cls == 0:
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    boxes = get_box_from_mask(p_mask, threshold=0.5)
                    if len(boxes) == 0:
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            box_strs.append(
                                f"opacity {p_conf:.6f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}"
                            )
                        image_pred_str = " ".join(box_strs)

                pred_rows.append(
                    {"id": f"{image_id}_image", "PredictionString": image_pred_str}
                )

    submission_df = pd.DataFrame(pred_rows)
    # Ensure unique IDs just in case
    submission_df = submission_df.drop_duplicates(subset=["id"])
    submission_df.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)
    print(f"Submission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")


def run(debug=False):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-4

    # Load Metadata
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    if debug:
        df_val = df_val.head(100)
        df_test = df_test.head(100)

    # Load Data
    train_ds = get_dataset("train", load_cached_data=True, debug=debug)
    val_ds = get_dataset("val", load_cached_data=True, debug=debug)
    test_ds = get_dataset("test", load_cached_data=True, debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = ResNet18UNet(num_classes=4, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch
    )

    scaler = GradScaler()

    best_map = 0.0

    print(f"Starting training on {device} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device
        )
        val_loss, val_map = evaluate(model, val_loader, device, df_val)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.6f}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  >>> New Best mAP! Model saved.")

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))

    predict_test(model, test_loader, device, df_test)
