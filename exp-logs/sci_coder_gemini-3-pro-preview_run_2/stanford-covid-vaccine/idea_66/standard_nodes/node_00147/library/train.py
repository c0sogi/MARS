import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_device, mcrmse
from library.data import process_data, RNADataset
from library.model import GCSDNModel


def mcrmse_loss_tensor(pred, target):
    """
    PyTorch implementation of MCRMSE loss for training.
    Calculates loss only on scored columns and scored sequence positions.
    """
    # pred, target: (B, 5, L)
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_cols = [0, 1, 3]
    scored_len = Config.SEQ_SCORED

    # Slice to valid region
    pred_scored = pred[:, scored_cols, :scored_len]
    target_scored = target[:, scored_cols, :scored_len]

    # MSE over batch and length
    mse = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 2))

    # RMSE per column
    rmse = torch.sqrt(mse)

    # Mean of RMSEs
    return torch.mean(rmse)


def train_one_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch with 2-pass iterative refinement.
    """
    model.train()
    total_loss = 0.0

    for features, p_idx, targets in loader:
        features = features.to(device)
        p_idx = p_idx.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Static Backbone
        z = model.forward_backbone(features)

        # Pass 1: Zero Initialization
        y_init = torch.zeros_like(targets)
        y1 = model.forward_pass(z, y_init, p_idx)

        # Pass 2: Feedback from Pass 1 (Detached to stop gradient loops)
        y2 = model.forward_pass(z, y1.detach(), p_idx)

        # Calculate Loss
        # L_total = L(y2) + 0.5 * L(y1)
        # We use strictly masked loss (only scored positions)
        loss1 = mcrmse_loss_tensor(y1, targets)
        loss2 = mcrmse_loss_tensor(y2, targets)
        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Validates the model using Global MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, p_idx, targets in loader:
            features = features.to(device)
            p_idx = p_idx.to(device)

            # Inference: Use Pass 2 output
            _, y2 = model(features, p_idx)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate global MCRMSE using the utility function
    score = mcrmse(all_preds, all_targets)
    return score


def train_model():
    """
    Main training routine.
    """
    set_seed(Config.SEED)
    device = get_device()

    # Load Data with Caching
    train_data = process_data(
        os.path.join(Config.METADATA_DIR, "train.csv"), cache_name=Config.CACHE_TRAIN
    )
    val_data = process_data(
        os.path.join(Config.METADATA_DIR, "val.csv"), cache_name=Config.CACHE_VAL
    )

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize Model
    model = GCSDNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {len(train_dataset)} samples...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_score}")


def generate_submission():
    """
    Generates the submission file using the best trained model.
    """
    set_seed(Config.SEED)
    device = get_device()

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("No model found. Run training first.")
        return

    # Load Model
    model = GCSDNModel().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Test Data
    test_data = process_data(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        is_test=True,
        cache_name=Config.CACHE_TEST,
    )
    test_dataset = RNADataset(test_data, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    preds_map = {}

    print("Generating predictions...")
    with torch.no_grad():
        for features, p_idx, ids in test_loader:
            features = features.to(device)
            p_idx = p_idx.to(device)

            # Use Pass 2 output for final prediction
            _, y2 = model(features, p_idx)
            y_np = y2.cpu().numpy()  # (B, 5, L)

            for i, sample_id in enumerate(ids):
                # Map predictions to id_seqpos
                for pos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{pos}"
                    preds_map[row_id] = y_np[i, :, pos]

    # Create Submission DataFrame
    print("Writing submission file...")
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Prepare data array
    # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []

    for _, row in sample_sub.iterrows():
        row_id = row["id_seqpos"]
        if row_id in preds_map:
            submission_data.append(preds_map[row_id])
        else:
            # Fallback (should not happen if test set matches)
            submission_data.append(np.zeros(5))

    submission_df = pd.DataFrame(
        submission_data,
        columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
    )
    submission_df.insert(0, "id_seqpos", sample_sub["id_seqpos"])

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
