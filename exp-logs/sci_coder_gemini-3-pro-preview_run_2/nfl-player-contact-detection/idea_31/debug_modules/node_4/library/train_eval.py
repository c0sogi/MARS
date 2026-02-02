import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, compute_mcc, optimize_threshold
from library.feature_engineering import FeatureEngineer
from library.dataset import DataProcessor
from library.models import SSERVN
from library.losses import FocalLoss


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        x_kin = batch["x_kin"].to(device)
        x_vis = batch["x_vis"].to(device)
        x_cat = batch["x_cat"].to(device)
        y = batch["y"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x_kin, x_vis, x_cat)

        # Compute loss
        loss = criterion(logits.view(-1), y)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * y.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a dataset. Returns average loss, probabilities, and targets.
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_kin = batch["x_kin"].to(device)
            x_vis = batch["x_vis"].to(device)
            x_cat = batch["x_cat"].to(device)
            y = batch["y"].to(device)

            logits = model(x_kin, x_vis, x_cat)
            loss = criterion(logits.view(-1), y)

            probs = torch.sigmoid(logits.view(-1))

            running_loss += loss.item() * y.size(0)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    return avg_loss, all_probs, all_targets


def load_validation_data(fe, load_cached=True):
    """
    Custom loader for validation data to bypass FeatureEngineer's hardcoded split logic.
    Reuses the granular processing methods of FeatureEngineer.
    """
    output_path = os.path.join(Config.WORKING_DIR, "val_features.parquet")

    if load_cached and os.path.exists(output_path):
        print(f"Loading final validation features from {output_path}")
        return pd.read_parquet(output_path)

    print("Generating validation features from scratch...")

    # 1. Load Validation Metadata
    df_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))

    if Config.DEBUG:
        print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} validation rows...")
        df_meta = df_meta.sample(
            min(len(df_meta), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        )

    relevant_gps = df_meta["game_play"].unique()

    # 2. Load and Process Tracking (from TRAIN source files)
    df_tracking_raw = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
    )
    df_tracking_raw = df_tracking_raw[
        df_tracking_raw["game_play"].isin(relevant_gps)
    ].copy()

    df_tracking_proc = fe.process_tracking_data(
        df_tracking_raw, load_cached=load_cached, cache_name="tracking_val"
    )
    del df_tracking_raw
    gc.collect()

    # 3. Load and Process Helmets (from TRAIN source files)
    df_helmets_raw = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "train_baseline_helmets.csv")
    )
    df_helmets_raw = df_helmets_raw[
        df_helmets_raw["game_play"].isin(relevant_gps)
    ].copy()

    df_helmets_proc = fe.process_helmet_data(
        df_helmets_raw, load_cached=load_cached, cache_name="helmets_val"
    )
    del df_helmets_raw
    gc.collect()

    # 4. Merge
    df_final = fe.merge_and_impute(df_meta, df_tracking_proc, df_helmets_proc)

    # 5. Save
    print(f"Saving final validation features to {output_path}")
    df_final.to_parquet(output_path, index=False)

    return df_final


def run_training():
    """
    Main function to orchestrate data loading, model training, and submission generation.
    """
    seed_everything(Config.SEED)
    Config.setup()
    Config.print_config()

    device = torch.device(Config.DEVICE)

    # ==========================
    # 1. Data Preparation
    # ==========================
    fe = FeatureEngineer()
    dp = DataProcessor()

    # --- Train Data ---
    df_train = fe.generate_features(split="train", load_cached=True)
    train_dataset = dp.get_dataset(df_train, split="train", fit_scalers=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    del df_train
    gc.collect()

    # --- Validation Data ---
    df_val = load_validation_data(fe, load_cached=True)
    val_dataset = dp.get_dataset(df_val, split="validation", fit_scalers=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    del df_val
    gc.collect()

    # ==========================
    # 2. Model Initialization
    # ==========================
    # Determine dimensions from dataset
    kin_dim = train_dataset.X_kin.shape[1]
    vis_dim = train_dataset.X_vis.shape[1]

    # Determine categorical cardinalities
    # X_cat is (N, 4). We need max index + 1 for each column.
    # Since we used the same encoders for P1 and P2, col 0&2 share cardinality, col 1&3 share cardinality.
    # However, simply taking max(col) + 1 is safe and effective.
    cat_cardinalities = []
    for i in range(4):
        max_idx = max(
            train_dataset.X_cat[:, i].max().item(), val_dataset.X_cat[:, i].max().item()
        )
        cat_cardinalities.append(max_idx + 1)

    print(f"Model Input Dims: Kinematic={kin_dim}, Visual={vis_dim}")
    print(f"Categorical Cardinalities: {cat_cardinalities}")

    model = SSERVN(kin_dim, vis_dim, cat_cardinalities).to(device)

    # ==========================
    # 3. Training Setup
    # ==========================
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    best_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("\nStarting Training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        # Optimize Threshold
        curr_thresh, curr_mcc = optimize_threshold(val_targets, val_probs, steps=100)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCC: {curr_mcc} | "
            f"Threshold: {curr_thresh}"
        )

        # Early Stopping Check
        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold = curr_thresh
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"\nTraining Complete. Best Val MCC: {best_mcc} at Threshold: {best_threshold}"
    )

    # ==========================
    # 4. Inference on Test Set
    # ==========================
    print("\nGenerating Test Predictions...")

    # Load Best Model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Load Test Data
    df_test = fe.generate_features(split="test", load_cached=True)
    test_dataset = dp.get_dataset(df_test, split="test", fit_scalers=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Predict
    _, test_probs, _ = evaluate(model, test_loader, criterion, device)

    # Apply Threshold
    test_preds = (test_probs >= best_threshold).astype(int)

    # Create Submission
    submission = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": test_preds}
    )

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(submission.head())
