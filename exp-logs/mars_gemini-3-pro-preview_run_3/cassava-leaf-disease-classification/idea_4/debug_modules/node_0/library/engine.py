import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import AverageMeter, get_score
from library.model import mixup_data, cutmix_data, mixup_criterion


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler):
    """
    Manages the training iteration for a single epoch using MixUp/CutMix and AMP.
    """
    model.train()
    losses = AverageMeter()

    accum_steps = Config.ACCUMULATION_STEPS
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # MixUp / CutMix Decision
        do_mixup = False
        do_cutmix = False
        p = np.random.rand()
        if p < Config.MIXUP_PROB:
            if np.random.rand() < 0.5:
                do_mixup = True
            else:
                do_cutmix = True

        with autocast():
            if do_mixup:
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, Config.MIXUP_ALPHA
                )
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif do_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, labels, Config.CUTMIX_ALPHA
                )
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Normalize loss for gradient accumulation
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.update(loss.item() * accum_steps, batch_size)

    print(f"Train Epoch: {epoch} | Loss: {losses.avg}")
    return losses.avg


def validate(epoch, model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set and prints metrics with full precision.
    """
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            acc = get_score(labels.cpu().numpy(), preds.cpu().numpy())

            losses.update(loss.item(), batch_size)
            scores.update(acc, batch_size)

    print(f"Valid Epoch: {epoch}")
    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Accuracy: {scores.avg}")

    return losses.avg, scores.avg


def inference_fn(model, test_loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass 1: Original
            out1 = model(images)

            # Forward pass 2: Horizontal Flip (TTA)
            if Config.USE_TTA:
                # Flip width dimension (dim 3 for BCHW)
                images_flip = torch.flip(images, dims=[3])
                out2 = model(images_flip)
                outputs = (out1 + out2) / 2.0
            else:
                outputs = out1

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds_list.extend(preds)

    return preds_list


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs
):
    """
    Runs the training loop for the specified number of epochs with Early Stopping.
    """
    scaler = GradScaler()
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(epoch, model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Time: {elapsed}")

        # Early Stopping & Model Checkpoint
        if val_acc > best_acc:
            print(f"Validation Accuracy Improved ({best_acc} ---> {val_acc})")
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in validation accuracy. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Accuracy: {best_acc}")


def generate_submission(model, test_loader, df_test, device):
    """
    Runs inference on the test set and saves the submission file.
    """
    print("Generating submission...")
    predictions = inference_fn(model, test_loader, device)

    df_test["label"] = predictions
    # Ensure only required columns are saved
    submission = df_test[["image_id", "label"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
