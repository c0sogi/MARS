import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.model import RNAModel
from library.data import get_dataloaders
from library.utils import set_seed, mcrmse


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    # Iterate over batches
    for batch in loader:
        # Move data to device
        sequence = batch["sequence"].to(device)
        loop = batch["loop"].to(device)
        distance = batch["distance"].to(device)
        targets = batch["target"].to(device)  # Shape: (B, 68, 3)

        optimizer.zero_grad()

        # Forward pass: (B, 107, 3)
        outputs = model(sequence, loop, distance)

        # Slice outputs to match target length (68) for loss calculation
        # The prompt specifies masked MSE on the first 68 positions.
        outputs_scored = outputs[:, : Config.PRED_LEN, :]

        # Compute Loss
        loss = criterion(outputs_scored, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * sequence.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            distance = batch["distance"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            outputs = model(sequence, loop, distance)

            # Slice to scored length
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored)
            all_targets.append(targets)

    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    score = mcrmse(y_true, y_pred)
    return score.item()


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            distance = batch["distance"].to(device)
            ids = batch["id"]

            # Forward pass: (B, 107, 3)
            outputs = model(sequence, loop, distance)
            outputs = outputs.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(outputs)

    # Concatenate predictions: (N_samples, 107, 3)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Columns required: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model predicts: reactivity (idx 0), deg_Mg_pH10 (idx 1), deg_Mg_50C (idx 2)
    # deg_pH10 and deg_50C should be 0.0

    submission_rows = []
    seq_len = Config.SEQ_LEN  # 107

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 3)

        for seqpos in range(seq_len):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    # Create DataFrame
    df_sub = pd.DataFrame(submission_rows)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = df_sub[cols]

    # Save
    print(f"Saving submission to {output_path}")
    df_sub.to_csv(output_path, index=False)


def run_training(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main execution function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=batch_size
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAModel(config=Config).to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.CLIP_GRAD
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  [New Best Model] Saved to {best_model_path}")

    print(f"Training Complete. Best Val MCRMSE: {best_mcrmse}")

    # 6. Generate Submission
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)


if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    run_training()
