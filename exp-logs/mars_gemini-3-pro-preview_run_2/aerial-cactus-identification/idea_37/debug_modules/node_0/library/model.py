import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed
from library.dataset import get_loaders
from library.model_components import UltraWideSERepNeXt


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits.view(-1), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, np.array(all_probs).flatten())
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits.view(-1), targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    val_loss = running_loss / len(loader.dataset)
    try:
        val_auc = roc_auc_score(all_targets, np.array(all_probs).flatten())
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def train_and_predict():
    """
    Main execution function:
    1. Loads data.
    2. Trains 5 models (one per seed) with Early Stopping.
    3. Performs inference using the ensemble and Test Time Augmentation (TTA).
    4. Saves the submission file.
    """
    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)
    device = torch.device(Config.DEVICE)

    # 2. Training Loop (Homogeneous Seed Averaging)
    for seed in Config.SEEDS:
        print(f"\n=== Starting Training for Seed {seed} ===")
        set_seed(seed)

        # Initialize Model in Training Mode (Multi-branch)
        model = UltraWideSERepNeXt(num_classes=Config.NUM_CLASSES, deploy=False)
        model = model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping Variables
        best_val_auc = -1.0
        patience_counter = 0
        best_checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_seed_{seed}.pth"
        )

        for epoch in range(Config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
                f"Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Cleanup
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # 3. Inference Loop (Ensemble + TTA)
    print("\n=== Starting Inference ===")

    # Load all trained models
    models = []
    for seed in Config.SEEDS:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for seed {seed}, skipping.")
            continue

        # Instantiate in training mode to load weights correctly
        model = UltraWideSERepNeXt(num_classes=Config.NUM_CLASSES, deploy=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        # Switch to inference mode (Structural Re-parameterization)
        # This fuses the multi-branch blocks into single conv layers
        model.switch_to_deploy()
        models.append(model)

    if not models:
        print("Error: No models available for inference.")
        return

    results = {}  # Dictionary to store id -> probability

    # Iterate over test set
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Test Time Augmentation (TTA) Views:
            # 1. Original
            # 2. Horizontal Flip
            # 3. Vertical Flip
            imgs_orig = images
            imgs_h = torch.flip(images, dims=[3])
            imgs_v = torch.flip(images, dims=[2])

            # Accumulate predictions from all models and all views
            batch_preds_sum = torch.zeros(images.size(0), device=device)
            count = 0

            for model in models:
                # Get probabilities for each view
                p_orig = torch.sigmoid(model(imgs_orig)).view(-1)
                p_h = torch.sigmoid(model(imgs_h)).view(-1)
                p_v = torch.sigmoid(model(imgs_v)).view(-1)

                # Sum them up
                batch_preds_sum += p_orig + p_h + p_v
                count += 3

            # Compute arithmetic mean
            batch_preds_avg = batch_preds_sum / count

            # Store results
            preds_np = batch_preds_avg.cpu().numpy()
            for img_id, pred in zip(ids, preds_np):
                results[img_id] = pred

    # 4. Save Submission
    submission_df = pd.DataFrame(
        [{"id": img_id, "has_cactus": prob} for img_id, prob in results.items()]
    )

    # Sort by ID to ensure consistent order
    submission_df = submission_df.sort_values("id")

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Run the pipeline
train_and_predict()
