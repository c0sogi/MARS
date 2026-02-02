import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.model import SS_DFRN, process_data, RNADataset


def train_one_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch using the Iterative Refinement strategy.
    """
    model.train()
    running_loss = 0.0

    for x, p_idx, y in loader:
        x = x.to(device)
        p_idx = p_idx.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # --- Pass 1: Zero Feedback ---
        # The model handles feedback=None internally by creating a zero tensor
        pred1 = model(x, p_idx, feedback=None)
        loss1 = mcrmse_loss(pred1, y)

        # --- Pass 2: Feedback from Pass 1 ---
        # Detach pred1 to stop gradients from flowing back into the first pass via feedback generation
        # The model handles the specific masking of feedback columns internally
        feedback_in = pred1.detach()
        pred2 = model(x, p_idx, feedback=feedback_in)
        loss2 = mcrmse_loss(pred2, y)

        # --- Combined Loss ---
        # Weighted sum as per strategy
        loss = (Config.LOSS_PASS2_WEIGHT * loss2) + (Config.LOSS_PASS1_WEIGHT * loss1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, p_idx, y in loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            y = y.to(device)

            # --- Pass 1 ---
            pred1 = model(x, p_idx, feedback=None)

            # --- Pass 2 ---
            # Use Pass 1 output as feedback for final prediction
            pred2 = model(x, p_idx, feedback=pred1)

            # Metric is calculated on the final output
            loss = mcrmse_loss(pred2, y)
            running_loss += loss.item()

    return running_loss / len(loader)


def train_model():
    """
    Main function to train the SS-DFRN model.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training on {device}...")

    # 1. Load and Process Data
    # process_data handles caching internally
    train_features, train_pidx, train_targets = process_data(
        Config.TRAIN_CSV, Config.TRAIN_CACHE, load_cached_data=True, is_test=False
    )
    val_features, val_pidx, val_targets = process_data(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data=True, is_test=False
    )

    # 2. Create Datasets and Loaders
    train_dataset = RNADataset(train_features, train_pidx, train_targets)
    val_dataset = RNADataset(val_features, val_pidx, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = SS_DFRN().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 5. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New Best Model Saved! Loss: {best_loss}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Loss: {best_loss}")


def predict_and_submit():
    """
    Generates predictions for the test set and creates a submission file.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print("Starting inference...")

    # 1. Load Test Data
    test_features, test_pidx, test_ids = process_data(
        Config.TEST_CSV, Config.TEST_CACHE, load_cached_data=True, is_test=True
    )

    test_dataset = RNADataset(test_features, test_pidx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Load Model
    model = SS_DFRN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model not found. Using random initialization.")

    model.eval()

    # 3. Generate Predictions
    all_preds = []

    with torch.no_grad():
        for x, p_idx in test_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)

            # Pass 1
            pred1 = model(x, p_idx, feedback=None)

            # Pass 2 (Final Prediction)
            pred2 = model(x, p_idx, feedback=pred1)

            all_preds.append(pred2.cpu().numpy())

    # Concatenate all batches: [Total_Samples, Seq_Len, 5]
    all_preds = np.concatenate(all_preds, axis=0)

    # 4. Format Submission
    print("Formatting submission...")
    submission_rows = []

    # Iterate over samples
    for i, sample_id in enumerate(test_ids):
        # The sequence length is 107
        seq_len = all_preds.shape[1]

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            vals = all_preds[i, pos, :]

            # Create row: [id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Orchestrates the training and submission pipeline.
    """
    train_model()
    predict_and_submit()
