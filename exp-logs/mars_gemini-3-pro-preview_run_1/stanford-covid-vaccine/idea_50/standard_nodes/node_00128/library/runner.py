import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import RNADataset
from library.model import StabilizedWideBiGRU
from library.loss_metric import masked_mse_loss, mcrmse


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, device):
    """Runs one epoch of training."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move data to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, loop, pair_indices)

        # Compute loss
        loss = masked_mse_loss(preds, targets, mask)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device):
    """Runs validation and computes MCRMSE."""
    model.eval()
    total_mcrmse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(seq, loop, pair_indices)

            # Compute metric
            score = mcrmse(preds, targets, mask)

            # Accumulate (metric is per batch, we average over batches)
            # Note: Ideally MCRMSE is computed globally, but averaging batch MCRMSE
            # is a standard approximation during training loop monitoring.
            total_mcrmse += score.item()
            num_batches += 1

    return total_mcrmse / num_batches if num_batches > 0 else 0.0


def train_model():
    """Main training routine."""
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = RNADataset(split="train", load_cached=True)
    val_dataset = RNADataset(split="val", load_cached=True)

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
    print("Initializing model...")
    model = StabilizedWideBiGRU().to(device)

    # 3. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mcrmse = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (MCRMSE: {best_mcrmse:.10f})")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse:.10f}")


def predict_and_submit():
    """Generates predictions for the test set and creates submission file."""
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Load Data
    print("Loading test dataset...")
    test_dataset = RNADataset(split="test", load_cached=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Model
    model = StabilizedWideBiGRU().to(device)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Model file not found at {best_model_path}. Run training first."
        )

    print(f"Loading model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 3. Inference
    all_preds = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]  # List of strings

            # (Batch, Seq_Len, 3)
            preds = model(seq, loop, pair_indices)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions: (N_samples, Seq_Len, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # 4. Format Submission
    print("Formatting submission...")

    submission_data = []

    # Target columns predicted by model
    pred_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # All columns required in submission
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(all_ids):
        # Shape: (Seq_Len, 3)
        sample_preds = all_preds[i]

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            # Get predicted values
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Unscored columns are 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = pd.DataFrame(submission_data, columns=columns)

    # 5. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    # This block is for local testing if run directly, but the prompt asks
    # for the module implementation. The functions above are the API.
    # To run the full pipeline:
    train_model()
    predict_and_submit()
