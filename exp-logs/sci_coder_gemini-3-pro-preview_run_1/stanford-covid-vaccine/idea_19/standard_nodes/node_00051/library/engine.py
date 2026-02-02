import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from library.config import Config
from library.model import RNAModel
from library.dataset import get_dataloader
from library.utils import set_seed, mcrmse


def masked_mse_loss(preds, targets):
    """
    Computes MSE loss only on the valid scored positions.

    Args:
        preds (torch.Tensor): Model predictions (Batch, Seq_Len, 3).
        targets (torch.Tensor): Ground truth (Batch, Pred_Len, 3).

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Slice predictions to match target length (68)
    # preds: (B, 107, 3) -> (B, 68, 3)
    preds_sliced = preds[:, : Config.PRED_LENGTH, :]

    # Compute MSE
    loss = nn.MSELoss()(preds_sliced, targets)
    return loss


def train_fn(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        seq, loop, dist, targets, _ = batch

        seq = seq.to(device)
        loop = loop.to(device)
        dist = dist.to(device)
        targets = targets.to(device)

        batch_size = seq.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(seq, loop, dist)

        # Compute Loss
        loss = masked_mse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            seq, loop, dist, targets, _ = batch

            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            outputs = model(seq, loop, dist)

            # Slice predictions to 68 for evaluation against ground truth
            outputs_sliced = outputs[:, : Config.PRED_LENGTH, :]

            all_preds.append(outputs_sliced.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse(all_targets, all_preds)
    return score


def run_training():
    # Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data Loaders
    train_loader = get_dataloader(
        mode="train", batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(mode="val", batch_size=Config.BATCH_SIZE, shuffle=False)

    # Model
    model = RNAModel()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    # T_max is total steps per epoch * epochs
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_score = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Early Stopping & Model Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_score})")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_score


def predict_and_submit():
    print("\nGenerating submission...")
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_loader = get_dataloader(
        mode="test", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Load Best Model
    model = RNAModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            seq, loop, dist, _, ids = batch

            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass: (B, 107, 3)
            outputs = model(seq, loop, dist)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate predictions: (N_test, 107, 3)
    preds_array = np.concatenate(preds_list, axis=0)

    # Prepare Submission Data
    # We need 5 columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model predicts: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)
    # We need to map these to the correct indices and fill others with 0.

    submission_data = []

    # Columns in submission: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_array[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LENGTH):
            # Row ID
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            # If seqpos >= 68, the model wasn't trained on it, but we output it anyway or 0.
            # To be safe and consistent with "unscored", we can zero out > 67,
            # but usually providing the model output is better if the structure is sound.
            # However, since we masked loss > 67, the output might be random.
            # Let's zero out positions > 67 to avoid noise.
            if seqpos < Config.PRED_LENGTH:
                reactivity = float(sample_preds[seqpos, 0])
                deg_Mg_pH10 = float(sample_preds[seqpos, 1])
                deg_Mg_50C = float(sample_preds[seqpos, 2])
            else:
                reactivity = 0.0
                deg_Mg_pH10 = 0.0
                deg_Mg_50C = 0.0

            # Unscored columns are always 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    # Create DataFrame
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_data, columns=cols)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {sub_df.shape}")


if __name__ == "__main__":
    # This block is for testing the module independently if needed,
    # but the main entry point will likely import run_training.
    pass
