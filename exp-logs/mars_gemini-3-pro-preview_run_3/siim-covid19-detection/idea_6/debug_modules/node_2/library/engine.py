import os
import time
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, calculate_map, collate_fn, seed_everything
from library.model import SwinCascadeRCNN
from library.dataset import CovidDataset
from library.transforms import get_transforms


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    model.train()

    loss_meter = AverageMeter()
    loss_rpn_box_meter = AverageMeter()
    loss_rpn_cls_meter = AverageMeter()
    loss_roi_box_meter = AverageMeter()
    loss_roi_cls_meter = AverageMeter()
    loss_study_meter = AverageMeter()

    start_time = time.time()

    for i, (images, targets, _) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # Aggregate losses
        # The model returns a dict of scalar tensors
        losses = sum(loss for loss in loss_dict.values())

        # Breakdown for logging (summing cascade stages if present)
        loss_rpn_box = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
        loss_rpn_cls = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
        loss_study = loss_dict.get("loss_study", torch.tensor(0.0)).item()

        # Sum cascade losses
        loss_roi_box = sum(
            v.item() for k, v in loss_dict.items() if "loss_cascade_box" in k
        )
        loss_roi_cls = sum(
            v.item() for k, v in loss_dict.items() if "loss_cascade_cls" in k
        )

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)
        optimizer.step()

        if scheduler is not None:
            # Step scheduler per iteration if it's a cyclic/cosine scheduler requiring it,
            # but Config says CosineAnnealingLR which is usually per epoch.
            # However, if using OneCycleLR it would be here.
            # Given Config.SCHEDULER logic isn't fully exposed in detail, we'll step per epoch in the main loop.
            pass

        # Update meters
        loss_meter.update(loss_value, len(images))
        loss_rpn_box_meter.update(loss_rpn_box, len(images))
        loss_rpn_cls_meter.update(loss_rpn_cls, len(images))
        loss_roi_box_meter.update(loss_roi_box, len(images))
        loss_roi_cls_meter.update(loss_roi_cls, len(images))
        loss_study_meter.update(loss_study, len(images))

    end_time = time.time()
    print(
        f"Epoch: [{epoch}] "
        f"Loss: {loss_meter.avg} "
        f"RPN_Box: {loss_rpn_box_meter.avg} "
        f"RPN_Cls: {loss_rpn_cls_meter.avg} "
        f"ROI_Box: {loss_roi_box_meter.avg} "
        f"ROI_Cls: {loss_roi_cls_meter.avg} "
        f"Study: {loss_study_meter.avg} "
        f"Time: {end_time - start_time}"
    )

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # Store predictions and targets for mAP calculation
    all_preds = []
    all_targets = []

    # Store study predictions for accuracy
    study_correct = 0
    study_total = 0

    for images, targets, _ in data_loader:
        images = images.to(device)
        # Targets are needed for metric calculation
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # outputs is a list of dicts: {'boxes', 'labels', 'scores', 'study_probs'}

        # 1. Collect Detection Results
        # Move to CPU for calculation
        for i, output in enumerate(outputs):
            pred_dict = {
                "boxes": output["boxes"].cpu(),
                "labels": output["labels"].cpu(),
                "scores": output["scores"].cpu(),
            }
            all_preds.append(pred_dict)

            target_dict = {
                "boxes": targets[i]["boxes"].cpu(),
                "labels": targets[i]["labels"].cpu(),
            }
            all_targets.append(target_dict)

            # 2. Collect Study Results
            study_prob = output["study_probs"]  # Tensor (4,)
            pred_label = torch.argmax(study_prob).item()
            true_label = targets[i]["study_label"].item()

            if pred_label == true_label:
                study_correct += 1
            study_total += 1

    # Calculate mAP
    # Config.NMS_IOU_THRESHOLD is used during inference inside model,
    # calculate_map uses a matching threshold (default 0.5)
    map_score = calculate_map(all_preds, all_targets, iou_threshold=0.5)

    study_acc = study_correct / study_total if study_total > 0 else 0.0

    print(f"Validation mAP: {map_score}")
    print(f"Validation Study Accuracy: {study_acc}")

    return map_score


def run(debug=False, epochs=None, batch_size=None):
    # 1. Setup Configuration
    Config.setup(debug=debug, epochs=epochs, batch_size=batch_size)
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Create Datasets and Dataloaders
    train_dataset = CovidDataset(
        df_train,
        transforms=get_transforms("train"),
        split="train",
        load_cached_data=True,
    )
    val_dataset = CovidDataset(
        df_val, transforms=get_transforms("valid"), split="val", load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Initialize Model
    print("Initializing Swin Cascade R-CNN...")
    model = SwinCascadeRCNN()
    model.to(device)

    # 5. Optimizer and Scheduler
    params = [p for p in model.parameters() if p.requires_grad]

    if Config.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(
            params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=Config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=Config.WEIGHT_DECAY,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 6. Training Loop
    best_map = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Step scheduler
        scheduler.step()

        # Evaluate
        val_map = evaluate(model, val_loader, device)

        # Early Stopping & Checkpointing
        if val_map > best_map:
            print(
                f"Validation mAP improved from {best_map} to {val_map}. Saving model..."
            )
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(
                f"Validation mAP did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best mAP: {best_map}")
