import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, mixup_data, mixup_criterion, get_score
from library.dataset import SETIDataset
from library.model import SiameseEfficientNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # inputs is a tuple: (on_target_tensor, off_target_tensor)
        on_target, off_target = inputs

        # Concatenate on and off streams along channel dimension to apply Mixup consistently
        # Shape becomes (Batch, 6, H, W)
        combined_inputs = torch.cat([on_target, off_target], dim=1)

        combined_inputs = combined_inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        mixed_inputs, y_a, y_b, lam = mixup_data(
            combined_inputs, targets, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Split back into Siamese branches
        # Config.IN_CHANNELS is 3
        mixed_on = mixed_inputs[:, : Config.IN_CHANNELS, :, :]
        mixed_off = mixed_inputs[:, Config.IN_CHANNELS :, :, :]

        optimizer.zero_grad()

        # Forward pass
        outputs = model((mixed_on, mixed_off))
        outputs = outputs.squeeze(1)

        # Compute Mixup loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * combined_inputs.size(0)
        dataset_size += combined_inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            on_target, off_target = inputs
            on_target = on_target.to(device)
            off_target = off_target.to(device)
            targets = targets.to(device)

            outputs = model((on_target, off_target))
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            dataset_size += targets.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    auc_score = get_score(all_targets, all_preds)

    return epoch_loss, auc_score


def inference(model, dataloader, device):
    """
    Performs inference with Test Time Augmentation (TTA).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            on_target, off_target = inputs
            on_target = on_target.to(device)
            off_target = off_target.to(device)

            batch_size = on_target.size(0)
            batch_preds = torch.zeros((batch_size,), device=device)

            # TTA Loop over 4 variations
            for tta_step in range(Config.TTA_STEPS):
                curr_on = on_target.clone()
                curr_off = off_target.clone()

                # Apply transformations
                if tta_step == 0:
                    # Original
                    pass
                elif tta_step == 1:
                    # Horizontal Flip (Time) -> Axis 3 (Width)
                    curr_on = torch.flip(curr_on, dims=[3])
                    curr_off = torch.flip(curr_off, dims=[3])
                elif tta_step == 2:
                    # Vertical Flip (Frequency) -> Axis 2 (Height)
                    curr_on = torch.flip(curr_on, dims=[2])
                    curr_off = torch.flip(curr_off, dims=[2])
                elif tta_step == 3:
                    # HV Flip
                    curr_on = torch.flip(curr_on, dims=[2, 3])
                    curr_off = torch.flip(curr_off, dims=[2, 3])

                outputs = model((curr_on, curr_off))
                probs = torch.sigmoid(outputs.squeeze(1))
                batch_preds += probs

            # Average predictions
            batch_preds /= Config.TTA_STEPS
            results.extend(batch_preds.cpu().numpy().tolist())

    return results


def run_training(debug=False):
    """
    Main training loop.
    """
    # Ensure directories exist
    Config.setup()
    seed_everything(Config.SEED)

    # Load Datasets
    train_dataset = SETIDataset(Config.TRAIN_METADATA, mode="train")
    val_dataset = SETIDataset(Config.VAL_METADATA, mode="val")

    if debug:
        # Use a small subset for debugging
        indices = list(range(min(len(train_dataset), 100)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

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

    # Setup Model
    device = torch.device(Config.DEVICE)
    model = SiameseEfficientNet()
    model = model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")

    print(f"Training complete. Best AUC: {best_auc}")


def generate_submission():
    """
    Generates submission file using the best trained model.
    """
    Config.setup()
    seed_everything(Config.SEED)

    # Load Test Data
    test_dataset = SETIDataset(Config.TEST_METADATA, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    device = torch.device(Config.DEVICE)
    model = SiameseEfficientNet()

    if not os.path.exists(Config.MODEL_PATH):
        print(
            f"Model file not found at {Config.MODEL_PATH}. Cannot generate submission."
        )
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model = model.to(device)

    print("Running inference on test set...")
    predictions = inference(model, test_loader, device)

    # Create Submission
    df_test = pd.read_csv(Config.TEST_METADATA)
    df_test["target"] = predictions

    # Save only required columns
    submission = df_test[["id", "target"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
