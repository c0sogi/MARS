import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_auc
from library.data import mixup_data, mixup_criterion
from library.model import WhaleConvNeXt


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device)
        targets = targets.to(device)

        # Apply Mixup
        data, targets_a, targets_b, lam = mixup_data(
            data, targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        outputs = model(data)
        # Squeeze outputs to match target shape (B, 1) -> (B,) if necessary
        outputs = outputs.squeeze(1)

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        count += data.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * data.size(0)
            count += data.size(0)

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / count

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc_score = calculate_auc(all_targets, all_preds)

    return avg_loss, auc_score


def fit_one_seed(train_loader, val_loader, seed, device):
    """
    Initializes and trains a model for a specific random seed.
    Implements Early Stopping and saves the best model.
    """
    print(f"Starting training for Seed {seed}...")

    # Initialize Model
    model = WhaleConvNeXt(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # Loss Function with Class Weighting
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    # Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = Config.get_cache_path(f"model_seed_{seed}.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_auc)

        print(
            f"Seed {seed} | Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Finished training for Seed {seed}. Best Val AUC: {best_val_auc}")
    return best_model_path, best_val_auc


def predict(model, loader, device):
    """
    Generates predictions for a given loader using a trained model.
    Returns a dictionary mapping clip_id to probability.
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for data, clip_ids in loader:
            data = data.to(device)

            outputs = model(data)
            outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs)

            probs_np = probs.cpu().numpy()

            for clip_id, prob in zip(clip_ids, probs_np):
                predictions[clip_id] = prob

    return predictions


def generate_submission(test_loader, model_paths, device):
    """
    Loads multiple models, performs ensemble averaging, and saves the submission file.
    """
    print(f"Generating submission using {len(model_paths)} models...")

    # Initialize dictionary to store accumulated probabilities
    accumulated_probs = {}

    # Initialize with 0
    # We need to scan the loader once to get all IDs if we want to pre-allocate,
    # but using a dict is safer and flexible.

    for i, model_path in enumerate(model_paths):
        print(
            f"Inference with model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}"
        )

        # Instantiate architecture
        model = WhaleConvNeXt(
            backbone_name=Config.BACKBONE,
            pretrained=False,  # No need to download weights, we load state_dict
            in_channels=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
        )

        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue

        model.to(device)

        # Get predictions
        preds = predict(model, test_loader, device)

        # Accumulate
        for clip_id, prob in preds.items():
            if clip_id not in accumulated_probs:
                accumulated_probs[clip_id] = 0.0
            accumulated_probs[clip_id] += prob

    # Average probabilities
    num_models = len(model_paths)
    final_preds = []

    for clip_id, total_prob in accumulated_probs.items():
        avg_prob = total_prob / num_models
        final_preds.append({"clip": clip_id, "probability": avg_prob})

    # Create DataFrame
    df_submission = pd.DataFrame(final_preds)

    # Ensure correct column order
    df_submission = df_submission[["clip", "probability"]]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    return df_submission
