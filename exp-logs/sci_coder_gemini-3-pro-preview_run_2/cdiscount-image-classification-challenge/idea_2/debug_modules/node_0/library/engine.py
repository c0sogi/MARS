import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library import config, utils


class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Loss function with Label Smoothing
        # PyTorch CrossEntropyLoss supports label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING).to(
            device
        )

        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler("cuda")

    def train_one_epoch(self, train_loader, epoch):
        batch_time = utils.AverageMeter("Time", ":6.3f")
        data_time = utils.AverageMeter("Data", ":6.3f")
        losses = utils.AverageMeter("Loss", ":.4e")
        top1 = utils.AverageMeter("Acc@1", ":6.2f")

        self.model.train()
        end = time.time()

        for i, (images, targets) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Mixup
            if config.USE_MIXUP:
                images, targets_a, targets_b, lam = utils.mixup_data(
                    images, targets, config.MIXUP_ALPHA
                )

            # Forward pass with AMP
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(images)

                if config.USE_MIXUP:
                    loss = utils.mixup_criterion(
                        self.criterion, output, targets_a, targets_b, lam
                    )
                else:
                    loss = self.criterion(output, targets)

            # Measure accuracy and record loss
            # For Mixup, we calculate accuracy against the primary target for logging purposes
            if config.USE_MIXUP:
                acc1 = utils.accuracy(output, targets_a, topk=(1,))[0]
            else:
                acc1 = utils.accuracy(output, targets, topk=(1,))[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step (OneCycleLR steps per batch)
            if self.scheduler is not None:
                self.scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
                )

        print(
            f"Epoch {epoch} finished. Avg Loss: {losses.avg:.6f}, Avg Acc: {top1.avg:.6f}"
        )
        return top1.avg, losses.avg

    def validate(self, val_loader):
        batch_time = utils.AverageMeter("Time", ":6.3f")
        top1 = utils.AverageMeter("Acc@1", ":6.2f")

        self.model.eval()
        end = time.time()

        print("Starting validation...")
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                # Input shape: (1, N, C, H, W) -> Squeeze to (N, C, H, W)
                # N is the number of images for this product
                images = images.squeeze(0).to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # Forward pass
                # We use float32 for validation stability
                output = self.model(images)

                # Late Fusion: Average Softmax Probabilities
                probs = torch.softmax(output, dim=1)
                avg_prob = torch.mean(
                    probs, dim=0, keepdim=True
                )  # Shape (1, Num_Classes)

                # Calculate accuracy
                acc1 = utils.accuracy(avg_prob, target, topk=(1,))[0]
                top1.update(acc1.item(), 1)  # Batch size is effectively 1 product

                batch_time.update(time.time() - end)
                end = time.time()

                if i % 1000 == 0:
                    print(
                        f"Test: [{i}/{len(val_loader)}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
                    )

        print(f"Validation Results - Acc@1: {top1.avg}")
        return top1.avg

    def fit(self, train_loader, val_loader, epochs, patience=3):
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            print(f"\nStarting Epoch {epoch}/{epochs}")

            # Train
            train_acc, train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_acc = self.validate(val_loader)

            print(
                f"Epoch {epoch} Summary: Train Acc={train_acc:.6f}, Val Acc={val_acc:.6f}"
            )

            # Checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience_counter = 0
                print(f"New best model found! Saving checkpoint.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best,
            )

            # Early Stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return best_acc


def inference(model, test_loader, device):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    model.eval()

    # Get mapping from index to category_id
    # The dataset in the loader has this mapping
    if hasattr(test_loader.dataset, "idx_to_cat"):
        idx_to_cat = test_loader.dataset.idx_to_cat
    else:
        # Fallback: Load category names and sort them to reconstruct mapping
        print("Warning: idx_to_cat not found in dataset. Reconstructing...")
        cat_df = pd.read_csv(config.CATEGORY_NAMES)
        unique_cats = sorted(cat_df["category_id"].unique())
        idx_to_cat = {i: cat for i, cat in enumerate(unique_cats)}

    results = []
    print("Starting inference on test set...")

    with torch.no_grad():
        for i, (images, product_id) in enumerate(test_loader):
            # Input shape: (1, N, C, H, W)
            images = images.squeeze(0).to(device)
            product_id = product_id.item()

            # Forward pass
            output = model(images)

            # Late Fusion: Average Softmax
            probs = torch.softmax(output, dim=1)
            avg_prob = torch.mean(probs, dim=0, keepdim=True)

            # Get prediction
            _, pred_idx = torch.max(avg_prob, 1)
            pred_idx = pred_idx.item()

            # Map to category_id
            pred_cat_id = idx_to_cat[pred_idx]

            results.append({"_id": product_id, "category_id": int(pred_cat_id)})

            if i % 5000 == 0:
                print(f"Processed {i}/{len(test_loader)} records...")

    # Save submission
    df_sub = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
    print(df_sub.head())
