import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, MCRMSELoss, get_global_rmse
from library.data import preprocess_data, RNADataset
from library.model import SSPFN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch with the 2-pass iterative refinement strategy.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, partner_indices, targets) in enumerate(loader):
        features = features.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # --- Pass 1: Zero Feedback ---
        # Initialize feedback with None (model handles this as zeros)
        pred1 = model(features, partner_indices, feedback_input=None)

        # --- Pass 2: Feedback from Pass 1 ---
        # Detach pred1 so gradients don't flow back through the feedback generation of Pass 1
        # The feedback loop is trained to correct the errors of the previous step
        feedback_input = pred1.detach()
        pred2 = model(features, partner_indices, feedback_input=feedback_input)

        # --- Loss Calculation ---
        # Loss on final prediction
        loss2 = criterion(pred2, targets)
        # Loss on intermediate prediction (auxiliary loss)
        loss1 = criterion(pred1, targets)

        # Weighted sum
        loss = loss2 + Config.PASS1_WEIGHT * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Validates the model using the 2-pass strategy and computes Global MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, partner_indices, targets in loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            # Pass 1
            pred1 = model(features, partner_indices, feedback_input=None)

            # Pass 2
            pred2 = model(features, partner_indices, feedback_input=pred1)

            # Collect results (move to CPU numpy)
            all_preds.append(pred2.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Global MCRMSE
    score = get_global_rmse(all_preds, all_targets)
    return score


def train_model(debug=False):
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Data
    # Using the Config paths for caching
    train_feats, train_p_idx, train_targets, train_ids = preprocess_data(
        Config.TRAIN_FILE, Config.TRAIN_CACHE, load_cached_data=True, is_test=False
    )
    val_feats, val_p_idx, val_targets, val_ids = preprocess_data(
        Config.VAL_FILE, Config.VAL_CACHE, load_cached_data=True, is_test=False
    )

    if debug:
        print("Debug mode: slicing data.")
        train_feats = train_feats[:100]
        train_p_idx = train_p_idx[:100]
        train_targets = train_targets[:100]
        train_ids = train_ids[:100]
        val_feats = val_feats[:20]
        val_p_idx = val_p_idx[:20]
        val_targets = val_targets[:20]
        val_ids = val_ids[:20]

    train_dataset = RNADataset(train_feats, train_p_idx, train_targets, train_ids)
    val_dataset = RNADataset(val_feats, val_p_idx, val_targets, val_ids)

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

    # 2. Initialize Model
    model = SSPFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = MCRMSELoss()

    # 3. Training Loop
    best_score = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, criterion, device)

        scheduler.step(val_score)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New best model saved! Score: {val_score}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")


def generate_submission():
    """
    Generates predictions for the test set and creates the submission file.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Load Test Data
    test_feats, test_p_idx, test_targets, test_ids = preprocess_data(
        Config.TEST_FILE, Config.TEST_CACHE, load_cached_data=True, is_test=True
    )

    test_dataset = RNADataset(test_feats, test_p_idx, test_targets, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Model
    model = SSPFN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model not found. Using initialized weights.")

    model.eval()

    # 3. Inference
    all_preds = []

    with torch.no_grad():
        for features, partner_indices, _ in test_loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            # Pass 1
            pred1 = model(features, partner_indices, feedback_input=None)

            # Pass 2
            pred2 = model(features, partner_indices, feedback_input=pred1)

            all_preds.append(pred2.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, 107, 5)

    # 4. Format Submission
    # We need to flatten the predictions to match sample_submission format
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []

    # Column order in output must match sample_submission
    # Config.ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # This matches the model output order.

    for i, sample_id in enumerate(test_ids):
        # Get predictions for this sample
        sample_preds = all_preds[i]  # (107, 5)

        # Iterate over sequence positions
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {
                "id_seqpos": row_id,
                "reactivity": row_preds[0],
                "deg_Mg_pH10": row_preds[1],
                "deg_pH10": row_preds[2],
                "deg_Mg_50C": row_preds[3],
                "deg_50C": row_preds[4],
            }
            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure column order
    cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
    submission_df = submission_df[cols]

    # Save
    Config.setup()  # Ensure dir exists
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(debug=False):
    train_model(debug=debug)
    generate_submission()


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing
    run_training()
