import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import AverageMeter, accuracy
from library.dataset import CdiscountDataset, collate_flatten
from library.model import get_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_idx_to_id_map():
    """
    Recreates the index to category_id mapping used in the Dataset.
    """
    if not os.path.exists(Config.CATEGORY_NAMES):
        raise FileNotFoundError(f"{Config.CATEGORY_NAMES} not found.")

    cats = pd.read_csv(Config.CATEGORY_NAMES)
    cats = cats.sort_values("category_id").reset_index(drop=True)

    # idx is the dataframe index, category_id is the value
    idx_to_id = cats["category_id"].to_dict()
    return idx_to_id


def train_one_epoch(
    train_loader, model, criterion, optimizer, scheduler, scaler, device, epoch
):
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    num_batches = len(train_loader)

    for i, (images, targets, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=True):
            output = model(images)
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Measure accuracy and record loss
        acc1 = accuracy(output, targets, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        if i % 500 == 0:
            print(
                f"Epoch: [{epoch}][{i}/{num_batches}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f}) "
                f"LR {scheduler.get_last_lr()[0]:.6f}"
            )

    return top1.avg, losses.avg


def evaluate(val_loader, model, criterion, device):
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=True):
                output = model(images)
                loss = criterion(output, targets)

            acc1 = accuracy(output, targets, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

    print(f" * Acc@1 {top1.avg} Loss {losses.avg}")
    return top1.avg, losses.avg


def inference(test_loader, model, device):
    """
    Generates predictions for the test set using Late Fusion.
    """
    model.eval()

    all_probs = []
    all_pids = []

    print("Starting inference...")

    with torch.no_grad():
        for i, (images, _, product_ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            with autocast(enabled=True):
                output = model(images)
                probs = torch.softmax(output, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_pids.append(product_ids.numpy())

            if i % 1000 == 0:
                print(f"Inference batch {i}/{len(test_loader)}")

    # Concatenate all results
    flat_probs = np.concatenate(all_probs, axis=0)
    flat_pids = np.concatenate(all_pids, axis=0)

    print(f"Inference complete. Aggregating {len(flat_pids)} image predictions...")

    # Efficient aggregation using numpy
    # Sort by product_id to group them
    sort_idx = np.argsort(flat_pids)
    sorted_pids = flat_pids[sort_idx]
    sorted_probs = flat_probs[sort_idx]

    # Find indices where product_id changes
    unique_pids, indices = np.unique(sorted_pids, return_index=True)

    # Sum probabilities for each product group
    # np.add.reduceat sums slices [indices[i]:indices[i+1]]
    summed_probs = np.add.reduceat(sorted_probs, indices, axis=0)

    # Get class index with max probability sum
    final_preds_idx = np.argmax(summed_probs, axis=1)

    # Map indices back to category_ids
    idx_to_id = get_idx_to_id_map()

    # Vectorized mapping using a lookup table
    max_idx = max(idx_to_id.keys())
    lookup_table = np.zeros(max_idx + 1, dtype=np.int64)
    for idx, cat_id in idx_to_id.items():
        lookup_table[idx] = cat_id

    final_category_ids = lookup_table[final_preds_idx]

    # Create submission DataFrame
    submission = pd.DataFrame({"_id": unique_pids, "category_id": final_category_ids})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data Loaders
    print("Initializing Datasets...")
    train_dataset = CdiscountDataset(Config.TRAIN_META, mode="train")
    val_dataset = CdiscountDataset(Config.VAL_META, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_flatten,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_flatten,
    )

    # Model
    print("Initializing Model...")
    model = get_model(pretrained=True)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LR,
        momentum=0.9,
        weight_decay=Config.WEIGHT_DECAY,
        nesterov=True,
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    scaler = GradScaler()

    best_acc = 0.0

    print("Starting Training...")
    for epoch in range(1, Config.EPOCHS + 1):
        train_acc, train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler, scaler, device, epoch
        )
        print(f"Epoch {epoch} Train Result: Acc {train_acc} Loss {train_loss}")

        val_acc, val_loss = evaluate(val_loader, model, criterion, device)
        print(f"Epoch {epoch} Val Result: Acc {val_acc} Loss {val_loss}")

        # Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }

        save_path = os.path.join(
            Config.MODEL_CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(state, save_path)

        if is_best:
            best_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "model_best.pth")
            torch.save(state, best_path)
            print(f"New best model saved with Acc {best_acc}")

    print("Training Complete.")


def run_inference():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Dataset
    print("Initializing Test Dataset...")
    test_dataset = CdiscountDataset(Config.TEST_META, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_flatten,
    )

    # Load Model
    print("Loading Best Model...")
    model = get_model(pretrained=False)

    best_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "model_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(
            f"Loaded model from epoch {checkpoint['epoch']} with acc {checkpoint['best_acc']}"
        )
    else:
        print("Warning: Best model not found. Using random weights.")

    model = model.to(device)

    # Run Inference
    inference(test_loader, model, device)


def main():
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    run_training()
    run_inference()
