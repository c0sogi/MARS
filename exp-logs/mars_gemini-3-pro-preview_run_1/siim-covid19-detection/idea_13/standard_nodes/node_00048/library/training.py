import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.dataset import SIIMDataset, get_transforms
from library.model import ResNet18_UNet
from library.loss import HybridLoss
from library.utils import mask2bbox, calculate_map


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        cls_logits, seg_logits = model(images)

        loss = criterion(cls_logits, seg_logits, labels, masks)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    # Step scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    return losses.avg


def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()

    # Containers for mAP calculation
    pred_boxes_all = []
    pred_scores_all = []
    pred_labels_all = []

    gt_boxes_all = []
    gt_labels_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            cls_logits, seg_logits = model(images)

            loss = criterion(cls_logits, seg_logits, labels, masks)
            losses.update(loss.item(), images.size(0))

            # --- Post-processing for mAP ---

            # Probabilities
            cls_probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

            # GT to numpy
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):
                # --- Ground Truth ---
                b_gt_boxes = []
                b_gt_labels = []

                # 1. Study Level GT: 1x1 box at 0,0 for the true label
                # Classes: 0:Neg, 1:Typ, 2:Ind, 3:Aty
                true_label = labels_np[i]
                b_gt_boxes.append([0, 0, 1, 1])
                b_gt_labels.append(true_label)

                # 2. Image Level GT: Boxes for Opacity (Class 4)
                gt_opacity_boxes = mask2bbox(masks_np[i], threshold=0.5)
                for box in gt_opacity_boxes:
                    b_gt_boxes.append(box)
                    b_gt_labels.append(4)  # Class 4 for Opacity

                gt_boxes_all.append(b_gt_boxes)
                gt_labels_all.append(b_gt_labels)

                # --- Predictions ---
                b_pred_boxes = []
                b_pred_scores = []
                b_pred_labels = []

                # 1. Study Level Predictions: 1x1 box for ALL classes with scores
                for cls_idx in range(4):
                    b_pred_boxes.append([0, 0, 1, 1])
                    b_pred_scores.append(float(cls_probs[i, cls_idx]))
                    b_pred_labels.append(cls_idx)

                # 2. Image Level Predictions: Opacity (Class 4)
                # Gating: If predicted study is Negative (0), force no opacity
                pred_study_label = np.argmax(cls_probs[i])

                if pred_study_label != 0:
                    # Extract boxes from predicted mask
                    pred_opacity_boxes = mask2bbox(seg_probs[i], threshold=0.5)

                    for box in pred_opacity_boxes:
                        x1, y1, x2, y2 = box
                        # Clamp coordinates
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(Config.IMG_SIZE, x2), min(Config.IMG_SIZE, y2)

                        # Score is mean probability within the box
                        box_mask = seg_probs[i, 0, y1:y2, x1:x2]
                        if box_mask.size > 0:
                            score = float(np.mean(box_mask))
                            b_pred_boxes.append(box)
                            b_pred_scores.append(score)
                            b_pred_labels.append(4)

                pred_boxes_all.append(b_pred_boxes)
                pred_scores_all.append(b_pred_scores)
                pred_labels_all.append(b_pred_labels)

    # Calculate mAP using the provided utility
    val_map = calculate_map(
        pred_boxes_all,
        pred_scores_all,
        pred_labels_all,
        gt_boxes_all,
        gt_labels_all,
        iou_threshold=0.5,
    )

    return losses.avg, val_map


def run_training():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # Initialize Datasets
    train_dataset = SIIMDataset(
        "train", load_cached_data=True, transform=get_transforms("train")
    )
    val_dataset = SIIMDataset(
        "val", load_cached_data=True, transform=get_transforms("val")
    )

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

    # Initialize Model
    model = ResNet18_UNet(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Initialize Loss
    criterion = HybridLoss().to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    best_map = 0.0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validation
        val_loss, val_map = valid_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mAP: {val_map:.6f}"
        )

        # Save Best Model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved! (mAP: {best_map:.6f})")

    print(f"Training Complete. Best Validation mAP: {best_map:.6f}")
