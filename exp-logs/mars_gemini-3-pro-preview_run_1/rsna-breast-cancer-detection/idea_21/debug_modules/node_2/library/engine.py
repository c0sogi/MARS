import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, setup_logger, probabilistic_f1
from library.model import SiameseEfficientNet
from library.data import get_loaders, BreastCancerDataset, get_transforms, get_age_stats


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (tgt, contra, label, _) in enumerate(loader):
        tgt = tgt.to(device)
        contra = contra.to(device)
        label = label.to(device)

        batch_size = tgt.size(0)

        optimizer.zero_grad()

        # Forward pass: Siamese network expects (target, contralateral)
        # Output shape is [B, 1], squeeze to [B] to match label
        logits = model(tgt, contra).squeeze(1)

        loss = criterion(logits, label)

        loss.backward()

        # Note: Gradient clipping is explicitly disabled as per requirements
        # to allow large updates for the minority class given the high pos_weight.
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for tgt, contra, label, _ in loader:
            tgt = tgt.to(device)
            contra = contra.to(device)
            label = label.to(device)
            batch_size = tgt.size(0)

            logits = model(tgt, contra).squeeze(1)
            loss = criterion(logits, label)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate Probabilistic F1
    pf1 = probabilistic_f1(all_labels, all_probs)

    return avg_loss, pf1


def run_training(debug=False, epochs=None):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    Config.setup(debug=debug, epochs=epochs)
    set_seed(Config.SEED)

    # Ensure working directory exists for logs
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "training.log"))

    device = torch.device(Config.DEVICE)
    logger.info(f"Device: {device}")

    # 2. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader = get_loaders(debug=debug, load_cached_data=True)

    # 3. Model Initialization
    logger.info(f"Initializing model: {Config.BACKBONE}")
    model = SiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=True)
    model = model.to(device)

    # 4. Optimization
    # Aggressive positive weighting for 1:47 imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training Loop
    best_pf1 = -1.0
    patience = 5
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging (Full precision)
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val pF1: {val_pf1}"
        )

        # Checkpointing & Early Stopping
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    logger.info(f"Training finished. Best Val pF1: {best_pf1}")


def generate_submission(debug=False):
    """
    Generates predictions for the test set and saves the submission file.
    """
    # 1. Setup
    Config.setup(debug=debug)
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Model
    model = SiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=False)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "Warning: No trained model found at specified path. Using random weights."
        )

    model = model.to(device)
    model.eval()

    # 3. Prepare Test Data
    # Load test metadata
    df_test = pd.read_csv(Config.TEST_CSV)
    if debug:
        df_test = df_test.head(100)

    # Get age stats (load from cache to ensure consistency with training)
    # We pass a dummy dataframe if needed, but get_age_stats mainly looks for cache first
    # If cache is missing, we would ideally need train data, but here we assume training ran first.
    # We will try to load cache, if fail, compute from test (suboptimal but prevents crash)
    try:
        age_mean, age_std = get_age_stats(df_test, load_cached_data=True)
    except Exception:
        # Fallback if training wasn't run and cache is missing
        age_mean, age_std = 58.7, 10.0

    test_dataset = BreastCancerDataset(
        df_test, transform=get_transforms("test"), age_mean=age_mean, age_std=age_std
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Inference
    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for tgt, contra, _, pred_ids in test_loader:
            tgt = tgt.to(device)
            contra = contra.to(device)

            logits = model(tgt, contra).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for pid, prob in zip(pred_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # 5. Aggregate and Save
    df_res = pd.DataFrame(results)

    # Group by prediction_id and take the MAX probability (as per task description strategy)
    # A prediction_id corresponds to a breast, which may have multiple views (CC, MLO)
    df_submission = df_res.groupby("prediction_id")["cancer"].max().reset_index()

    out_path = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
    df_submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
