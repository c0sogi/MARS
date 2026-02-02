import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import set_seed, AverageMeter
from library.dataset import INatDataset, get_transforms
from library.model import get_model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(
    train_loader, model, criterion, optimizer, device, epoch, scaler=None
):
    """
    Trains the model for one epoch using AMP if scaler is provided.
    Cite solution_lesson_node_00006: Use Mixed Precision (AMP).
    """
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Measure accuracy and record loss
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # Backward pass and optimize
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    print(f"Epoch [{epoch}] Training Loss: {losses.avg}, Training Acc: {top1.avg}")
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    print(
        f"Validation Results - Loss: {losses.avg}, Top-1 Acc: {top1.avg}, Top-5 Acc: {top5.avg}"
    )
    return top1.avg


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()

    test_dataset = INatDataset(
        csv_path=Config.TEST_CSV, mode="test", transform=get_transforms(stage="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    predictions = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Get top 5 predictions
            _, top5_indices = torch.topk(outputs, k=5, dim=1)

            top5_indices = top5_indices.cpu().numpy()
            # Handle image_ids whether they are tensors or lists
            if isinstance(image_ids, torch.Tensor):
                image_ids = image_ids.numpy()

            for img_id, preds in zip(image_ids, top5_indices):
                pred_str = " ".join(map(str, preds))
                predictions.append({"id": img_id, "predicted": pred_str})

    df = pd.DataFrame(predictions)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Main training loop orchestration.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data Loading
    train_dataset = INatDataset(
        csv_path=Config.TRAIN_CSV, mode="train", transform=get_transforms(stage="train")
    )

    val_dataset = INatDataset(
        csv_path=Config.VAL_CSV, mode="val", transform=get_transforms(stage="val")
    )

    if debug:
        # Subset for debugging
        print("Debug mode: Reducing dataset size for quick check.")
        indices_train = list(range(min(len(train_dataset), 1000)))
        indices_val = list(range(min(len(val_dataset), 200)))
        train_dataset = Subset(train_dataset, indices_train)
        val_dataset = Subset(val_dataset, indices_val)

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

    # Model Setup
    model = get_model(
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        device=Config.DEVICE,
    )

    # Cite solution_lesson_node_00006: Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Cite solution_lesson_node_00006: Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, epochs + 1):
        print(f"\nStarting Epoch {epoch}/{epochs}")

        # Train
        train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, scaler
        )

        # Validate
        val_acc = validate(val_loader, model, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Checkpointing and Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for submission
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    # Generate Submission
    generate_submission(model, device)
