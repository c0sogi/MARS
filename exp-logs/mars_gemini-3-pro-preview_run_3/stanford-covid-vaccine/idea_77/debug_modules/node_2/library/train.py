import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from library.config import Config
from library.data import get_dataloaders, get_test_dataloader
from library.model import RNAModel
from library.utils import seed_everything, MCRMSE


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_indices, pair_mask)

        # Calculate loss (MCRMSE on all 5 columns, sliced to 68)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, scorer, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, pair_indices, pair_mask)

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate all batches
    if not all_preds:
        return float("inf")

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metric (MCRMSE on specific scored columns, sliced to 68)
    metric = scorer(all_preds, all_targets)
    return metric.item()


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns numpy array of shape (N, 107, 5).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)

            preds = model(inputs, pair_indices, pair_mask)
            all_preds.append(preds.cpu().numpy())

    if not all_preds:
        return np.array([])

    return np.concatenate(all_preds, axis=0)


def run_training(
    max_epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    max_train_samples=None,
    max_val_samples=None,
):
    """
    Main training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    # 3. Model
    model = RNAModel(Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # 5. Metrics
    # Training loss: All 5 columns, sliced to 68
    train_criterion = MCRMSE(pred_len=Config.PRED_LEN, scored_indices=None)
    # Validation metric: Specific 3 columns (0, 1, 3), sliced to 68
    # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
    val_scorer = MCRMSE(pred_len=Config.PRED_LEN, scored_indices=[0, 1, 3])

    # 6. Loop
    best_metric = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {max_epochs} epochs...")

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion, device
        )
        val_metric = validate(model, val_loader, val_scorer, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_metric:.10f}"
        )

        # Save best model
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print("  New best model saved!")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_model_path


def generate_submission(model_path, output_path=Config.SUBMISSION_PATH):
    """
    Generates submission file using the trained model.
    """
    print("Generating submission...")
    device = Config.DEVICE

    # Load Model
    model = RNAModel(Config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load Test Data
    test_loader = get_test_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Predict
    # Shape: (N_test, 107, 5)
    preds = predict(model, test_loader, device)

    # Get Test IDs
    # We need to read the test file again to get IDs because dataloader might not expose them easily
    # in the order we need if we didn't store them.
    # However, RNADataset in config.py supports 'ids'.
    # The get_test_dataloader function in data.py loads IDs and passes them to dataset.
    # But the predict function just returns arrays.
    # We can read the test parquet file to get IDs in order.
    test_df = pd.read_parquet(Config.TEST_PATH)
    test_ids = test_df["id"].values

    if len(preds) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(preds)}) does not match number of test IDs ({len(test_ids)})."
        )

    # Format Submission
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_vals[col_idx])

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# Note: The __name__ == "__main__" block is omitted as per instructions.
# The functions above can be imported and run by a driver script.
