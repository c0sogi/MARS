import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

from library.config import Config
from library.utils import AverageMeter, Mixup, accuracy, save_checkpoint
from library.dataset import get_category_mapping, collate_flatten, collate_product


def train_one_epoch(
    train_loader, model, criterion, optimizer, scheduler, epoch, device, mixup_fn
):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_AMP)

    # Iterate over data
    # Note: train_loader uses collate_flatten, so inputs are (N, C, H, W) and targets are (N,)
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_fn(images, targets)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
            outputs = model(images)
            loss = Mixup.criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Measure accuracy and record loss
        # For accuracy during mixup, we can just compare against the dominant target or skip
        # Here we calculate accuracy against the first target (y_a) weighted by lam for approximation,
        # or just skip detailed acc logging during training to save time/complexity,
        # but the prompt asks for metrics. We'll use y_a.
        prec1 = accuracy(outputs, targets_a, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1[0].item(), images.size(0))

    print(f"Epoch: [{epoch}] Training Loss: {losses.avg} Training Acc: {top1.avg}")
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set using Late Fusion.
    """
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        # val_loader uses collate_product
        # batch structure: (all_images, all_pids, all_labels, sizes)
        for images, pids, targets, sizes in val_loader:
            images = images.to(device, non_blocking=True)
            # targets in the batch are repeated per image, we need one per product.
            # We can extract them based on the cumulative sum of sizes or just take the first one per group.
            # However, `collate_product` returns a flat tensor of labels corresponding to images.
            # We need to reduce this to one label per product.

            # Since all images of a product share the label, we can just take the label at the start index of each group.
            # But to be safe and use the tensor structure:

            with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                outputs = model(images)  # (Total_Images, Num_Classes)

            # Split outputs by product
            split_outputs = torch.split(outputs, sizes.tolist())

            # Aggregate predictions (Mean pooling of logits/probs)
            # Using logits average is standard for late fusion
            product_preds = []
            product_targets = []

            current_idx = 0
            targets_cpu = targets.cpu().numpy()

            for i, size in enumerate(sizes):
                # Average logits for this product
                prod_output = split_outputs[i].mean(dim=0)
                product_preds.append(prod_output)

                # Get the target for this product (all images have same target)
                # targets tensor corresponds to images, so we take the one at current_idx
                product_targets.append(targets[current_idx])
                current_idx += size.item()

            product_preds = torch.stack(product_preds)  # (Batch_Size, Num_Classes)
            product_targets = torch.stack(product_targets).to(device)  # (Batch_Size,)

            # Calculate Loss (Optional for Val, but good metric)
            loss = criterion(product_preds, product_targets)

            # Calculate Accuracy
            prec1 = accuracy(product_preds, product_targets, topk=(1,))

            losses.update(loss.item(), len(product_targets))
            top1.update(prec1[0].item(), len(product_targets))

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Accuracy: {top1.avg}")
    return top1.avg


def inference(test_loader, model, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()

    # Load category mapping to invert it
    cat_map = get_category_mapping(load_cached_data=True)
    inv_map = {v: k for k, v in cat_map.items()}

    results = []

    print("Starting Inference...")

    with torch.no_grad():
        for images, pids, _, sizes in test_loader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                outputs = model(images)

            split_outputs = torch.split(outputs, sizes.tolist())

            # We need to extract the product IDs corresponding to the groups
            # pids tensor in collate_product is repeated per image.
            # We need one pid per group.
            pids_cpu = pids.numpy()
            current_idx = 0

            for i, size in enumerate(sizes):
                # Average logits
                prod_output = split_outputs[i].mean(dim=0)

                # Get predicted class index
                _, pred_idx = prod_output.topk(1, 0, True, True)
                pred_idx = pred_idx.item()

                # Map back to category_id
                pred_cat_id = inv_map.get(pred_idx, -1)  # Should not fail

                # Get product ID
                prod_id = pids_cpu[current_idx]

                results.append({"_id": int(prod_id), "category_id": int(pred_cat_id)})

                current_idx += size.item()

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved.")


def fit(model, train_loader, val_loader, test_loader, epochs, device):
    """
    Main training loop with Early Stopping.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LR_MAX,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR_MAX,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA)

    best_acc = 0.0
    patience = 3  # Early stopping patience
    patience_counter = 0

    # Attempt to load checkpoint if exists (resuming)
    start_epoch = 0
    # Logic to load checkpoint could go here, but we start fresh for this task usually

    for epoch in range(start_epoch, epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            device,
            mixup_fn,
        )

        # Validate
        val_acc = validate(val_loader, model, criterion, device)

        # Checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

        print(f"Best Accuracy so far: {best_acc}")

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    # After training, load best model for inference
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for inference...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # Inference
    inference(test_loader, model, device)
