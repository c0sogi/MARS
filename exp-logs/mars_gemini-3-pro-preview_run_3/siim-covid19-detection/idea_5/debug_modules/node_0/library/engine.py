import math
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, calculate_map
from library.dataset import CovidDataset, collate_fn
from library.model import get_model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()

    header = f"Epoch: [{epoch}]"

    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    running_loss_study = 0.0

    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # Apply weights from Config
        # Note: loss_study is already weighted inside the model class
        losses = (
            loss_dict["loss_classifier"] * Config.LOSS_WEIGHT_ROI_CLS
            + loss_dict["loss_box_reg"] * Config.LOSS_WEIGHT_ROI_BOX
            + loss_dict["loss_objectness"] * Config.LOSS_WEIGHT_RPN_CLS
            + loss_dict["loss_rpn_box_reg"] * Config.LOSS_WEIGHT_RPN_BOX
            + loss_dict["loss_study"]
        )

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update running metrics
        running_loss += loss_value
        running_loss_classifier += loss_dict["loss_classifier"].item()
        running_loss_box_reg += loss_dict["loss_box_reg"].item()
        running_loss_objectness += loss_dict["loss_objectness"].item()
        running_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()
        running_loss_study += loss_dict["loss_study"].item()

        if i % print_freq == 0:
            print(
                f"{header} [{i}/{len(data_loader)}] "
                f"Loss: {loss_value:.4f} "
                f"(Cls: {loss_dict['loss_classifier'].item():.4f}, "
                f"Box: {loss_dict['loss_box_reg'].item():.4f}, "
                f"Obj: {loss_dict['loss_objectness'].item():.4f}, "
                f"RPN: {loss_dict['loss_rpn_box_reg'].item():.4f}, "
                f"Study: {loss_dict['loss_study'].item():.4f})"
            )

    avg_loss = running_loss / len(data_loader)
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # Containers for mAP calculation
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []

    gt_boxes_list = []
    gt_labels_list = []

    # Containers for Study Accuracy
    study_correct = 0
    study_total = 0

    print("Running validation...")

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        # targets are needed for GT comparison

        # Inference
        outputs = model(images)

        # Process batch
        for i, output in enumerate(outputs):
            target = targets[i]

            # --- Object Detection Metrics ---
            # Predictions
            boxes = output["boxes"].cpu().numpy().tolist()
            scores = output["scores"].cpu().numpy().tolist()
            labels = output["labels"].cpu().numpy().tolist()

            pred_boxes_list.append(boxes)
            pred_scores_list.append(scores)
            pred_labels_list.append(labels)

            # Ground Truth
            gt_boxes = target["boxes"].numpy().tolist()
            gt_labels = target["labels"].numpy().tolist()

            gt_boxes_list.append(gt_boxes)
            gt_labels_list.append(gt_labels)

            # --- Study Classification Metrics ---
            # Predicted study label
            pred_study_label = output["study_label"].item()
            gt_study_label = target["study_ids"].item()

            if pred_study_label == gt_study_label:
                study_correct += 1
            study_total += 1

    # Calculate mAP
    # We calculate mAP at IoU=0.5 as per the task metric
    map_score = calculate_map(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        iou_threshold=0.5,
    )

    # Calculate Study Accuracy
    study_acc = study_correct / study_total if study_total > 0 else 0.0

    print(f"Validation Results - mAP@0.5: {map_score}, Study Accuracy: {study_acc}")

    return map_score, study_acc


def train_model(num_epochs=Config.NUM_EPOCHS, save_path=Config.MODEL_SAVE_PATH):
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing model on {device}...")
    model = get_model()
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )

    # Datasets
    print("Loading datasets...")
    train_dataset = CovidDataset(mode="train", load_cached_data=True)
    val_dataset = CovidDataset(mode="val", load_cached_data=True)

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

    best_map = 0.0
    patience = 3
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Step Scheduler
        lr_scheduler.step()

        # Validate
        val_map, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch} Summary: Avg Loss: {avg_loss}, mAP: {val_map}, Study Acc: {val_acc}"
        )

        # Save Best Model
        if val_map > best_map:
            print(f"mAP improved from {best_map} to {val_map}. Saving model...")
            best_map = val_map
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in mAP. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best mAP: {best_map}")
