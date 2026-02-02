import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, compute_dice_score, rle_encode
from library.model import ShuffleNetPSPNet
from library.dataset import get_dataloaders
from library.loss import CombinedLoss
from library.post_processing import refine_predictions


def poly_lr_scheduler(optimizer, current_iter, max_iter, lr_start, power=0.9):
    """
    Adjusts the learning rate according to the polynomial decay policy.
    lr = lr_start * (1 - iter/max_iter)^power
    """
    if current_iter > max_iter:
        return

    lr = lr_start * ((1 - float(current_iter) / max_iter) ** power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, epoch, total_epochs, start_iter
):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    max_iter = total_epochs * len(loader)

    for i, batch in enumerate(loader):
        current_iter = start_iter + i

        # Update learning rate
        poly_lr_scheduler(
            optimizer, current_iter, max_iter, Config.LEARNING_RATE, Config.POLY_POWER
        )

        # Move data to device
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward and Backward
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss, start_iter + len(loader)


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            # Compute Dice for monitoring (Hard threshold)
            preds = torch.sigmoid(outputs) > Config.THRESHOLD
            preds = preds.cpu().numpy().astype(np.uint8)
            targets = masks.cpu().numpy().astype(np.uint8)

            # compute_dice_score calculates global overlap for the flattened arrays
            batch_dice = compute_dice_score(preds, targets)
            dice_scores.append(batch_dice)

    return running_loss / len(loader), np.mean(dice_scores)


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set, applies post-processing, and saves to CSV.
    """
    print("Generating submission...")
    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            ids = batch["id"]  # List of IDs

            outputs = model(images)
            preds = torch.sigmoid(outputs) > Config.THRESHOLD
            preds = preds.cpu().numpy().astype(np.uint8)

            # Iterate over batch
            for i, img_id in enumerate(ids):
                # Iterate over classes
                for class_idx, class_name in enumerate(Config.CLASS_LABELS):
                    mask = preds[i, class_idx]
                    rle = rle_encode(mask)
                    results.append(
                        {"id": img_id, "class": class_name, "predicted": rle}
                    )

    # Create initial DataFrame
    pred_df = pd.DataFrame(results)

    # Load test metadata for refinement (dimensions, case info)
    if os.path.exists(Config.TEST_METADATA_PATH):
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

        # Apply 3D Post-Processing (Largest Connected Component)
        final_df = refine_predictions(pred_df, test_meta)
    else:
        print("Warning: Test metadata not found. Skipping 3D refinement.")
        final_df = pred_df

    # Save submission
    Config.setup()  # Ensure directories exist
    final_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.
    """
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training on {device}...")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data, debug=debug
    )

    # Initialize Model
    model = ShuffleNetPSPNet(
        num_classes=Config.NUM_CLASSES, in_channels=Config.IN_CHANNELS
    )
    model = model.to(device)

    # Optimizer and Loss
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )
    loss_fn = CombinedLoss().to(device)

    # Training State
    best_dice = 0.0
    patience = 5
    patience_counter = 0
    global_iter = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, global_iter = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, epochs, global_iter
        )

        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.6f}"
        )

        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with Dice: {best_dice:.6f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Training finished. Best Validation Dice: {best_dice:.6f}")

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, test_loader, device)
