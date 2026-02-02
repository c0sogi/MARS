import math
import sys
import time
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    LR_DECAY_STEP,
    LR_GAMMA,
    GRADIENT_CLIP_NORM,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    SEED,
)
from library.utils import (
    collate_fn,
    seed_everything,
    get_train_transforms,
    get_valid_transforms,
)
from library.dataset import CovidDataset
from library.model import get_model


class AverageMeter(object):
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


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()

    # Trackers for different loss components
    loss_meter = AverageMeter()
    loss_rpn_meter = AverageMeter()
    loss_roi_meter = AverageMeter()
    loss_mil_meter = AverageMeter()

    header = f"Epoch: [{epoch}]"

    for i, (images, targets, _) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Aggregate losses
        # loss_dict keys: 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg', 'loss_mil'

        # Grouping for logging clarity
        rpn_losses = sum(
            loss for k, loss in loss_dict.items() if "rpn" in k or "objectness" in k
        )
        roi_losses = sum(
            loss for k, loss in loss_dict.items() if "classifier" in k or "box_reg" in k
        )
        mil_loss = loss_dict.get("loss_mil", torch.tensor(0.0, device=device))

        losses = sum(loss for loss in loss_dict.values())

        # Check for infinity
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # Gradient Clipping
        utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

        optimizer.step()

        # Update meters
        loss_meter.update(loss_value)
        loss_rpn_meter.update(rpn_losses.item())
        loss_roi_meter.update(roi_losses.item())
        loss_mil_meter.update(mil_loss.item())

        if i % print_freq == 0:
            print(
                f"{header} [{i}/{len(data_loader)}] "
                f"Loss: {loss_meter.val:.6f} ({loss_meter.avg:.6f}) "
                f"RPN: {loss_rpn_meter.val:.6f} ({loss_rpn_meter.avg:.6f}) "
                f"ROI: {loss_roi_meter.val:.6f} ({loss_roi_meter.avg:.6f}) "
                f"MIL: {loss_mil_meter.val:.6f} ({loss_mil_meter.avg:.6f})"
            )

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, data_loader, device):
    # To calculate validation loss, we must use model.train() mode.
    # Standard Faster R-CNN returns detections in eval() mode, not losses.
    # We use torch.no_grad() to ensure no gradients are computed.
    model.train()

    loss_meter = AverageMeter()

    print("Evaluating validation loss...")
    for images, targets, _ in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_meter.update(losses.item())

    print(f"Validation Loss: {loss_meter.avg}")
    return loss_meter.avg


def fit(load_cached_data=True, debug=False):
    """
    Main training loop.
    """
    seed_everything(SEED)

    # 1. Data Preparation
    print("Loading datasets...")
    train_dataset = CovidDataset(
        mode="train",
        transforms=get_train_transforms(),
        load_cached_data=load_cached_data,
        debug=debug,
    )
    val_dataset = CovidDataset(
        mode="val",
        transforms=get_valid_transforms(),
        load_cached_data=load_cached_data,
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Model Setup
    print("Initializing model...")
    model = get_model()
    model.to(DEVICE)

    # 3. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_STEP, gamma=LR_GAMMA
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    print(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        # Step Scheduler
        lr_scheduler.step()

        # Validate
        val_loss = evaluate(model, val_loader, DEVICE)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch} completed in {epoch_time:.0f}s. "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = f"{WORKING_DIR}/best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")
