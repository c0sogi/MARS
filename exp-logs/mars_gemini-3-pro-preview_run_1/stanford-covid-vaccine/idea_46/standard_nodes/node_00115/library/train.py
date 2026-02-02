import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    # Only train on the first 68 positions
    scored_len = Config.SCORED_SEQ_LENGTH

    for batch in dataloader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        structure_dist = batch["structure_dist"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequence, loop_type, structure_dist)

        # Slice to scored length for loss calculation
        # outputs shape: (Batch, 107, 3) -> (Batch, 68, 3)
        outputs_scored = outputs[:, :scored_len, :]
        targets_scored = targets[:, :scored_len, :]

        # Compute loss
        loss = criterion(outputs_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()

    all_preds = []
    all_targets = []
    scored_len = Config.SCORED_SEQ_LENGTH

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            structure_dist = batch["structure_dist"].to(device)
            targets = batch["target"].to(device)

            outputs = model(sequence, loop_type, structure_dist)

            # Slice to scored length
            outputs_scored = outputs[:, :scored_len, :]
            targets_scored = targets[:, :scored_len, :]

            all_preds.append(outputs_scored.cpu())
            all_targets.append(targets_scored.cpu())

    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Compute MCRMSE
    score = mcrmse_loss(y_true, y_pred)

    return score.item()


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            structure_dist = batch["structure_dist"].to(device)
            ids = batch["id"]

            # Forward pass (full length 107)
            outputs = model(sequence, loop_type, structure_dist)

            # outputs shape: (Batch, 107, 3)
            preds_np = outputs.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds_np)

    # Concatenate all predictions: (N_samples, 107, 3)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    submission_data = []

    # Columns required: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model outputs: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns with 0.0
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

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=batch_size, debug=debug
    )

    # Initialize Model
    model = RNAModel(Config).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_mcrmse = evaluate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with MCRMSE: {best_mcrmse}")

    print(f"Training complete. Best MCRMSE: {best_mcrmse}")

    # Load best model for submission
    print("Loading best model for submission generation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    # This block is provided for testing purposes if run directly,
    # but the functions are designed to be imported.
    run_training()
