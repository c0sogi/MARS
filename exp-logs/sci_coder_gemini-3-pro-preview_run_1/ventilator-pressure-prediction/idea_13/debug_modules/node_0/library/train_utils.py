import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from library.config import Config
from library.model import VentilatorModel
from library.data_utils import load_test_ids


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss masked by the inspiratory phase (u_out == 0).
    Combines final prediction loss with auxiliary loss.
    """

    def __init__(self, aux_weight=0.3):
        super().__init__()
        self.aux_weight = aux_weight
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, target, u_out):
        """
        Args:
            preds: Tuple (final_pred, aux_pred)
                final_pred: (batch, seq_len, 1)
                aux_pred: (batch, seq_len, 1) or None
            target: (batch, seq_len)
            u_out: (batch, seq_len) - Binary control signal (1=expiratory, 0=inspiratory)
        """
        final_pred, aux_pred = preds

        # Squeeze predictions to match target shape
        final_pred = final_pred.squeeze(-1)

        # Create mask: 1 for inspiratory (u_out=0), 0 for expiratory (u_out=1)
        mask = 1 - u_out

        # Calculate pixel-wise L1 loss
        loss_final = self.l1(final_pred, target)

        # Apply mask
        masked_loss_final = loss_final * mask

        # Normalize by the number of valid time steps
        # Add epsilon to avoid division by zero
        sum_mask = mask.sum() + 1e-8
        term_final = masked_loss_final.sum() / sum_mask

        total_loss = term_final

        # Auxiliary Loss
        if aux_pred is not None:
            aux_pred = aux_pred.squeeze(-1)
            loss_aux = self.l1(aux_pred, target)
            masked_loss_aux = loss_aux * mask
            term_aux = masked_loss_aux.sum() / sum_mask

            total_loss += self.aux_weight * term_aux

        return total_loss


def train_epoch(model, loader, optimizer, scheduler, device, loss_fn, u_out_idx):
    """
    Runs one training epoch.
    """
    model.train()
    running_loss = 0.0

    # Iterate without progress bar to keep output clean as per requirements
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Extract u_out for masking
        # inputs shape: (batch, seq_len, features)
        u_out = inputs[:, :, u_out_idx]

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        # Calculate loss
        loss = loss_fn(preds, targets, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        optimizer.step()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device, u_out_idx):
    """
    Evaluates the model on the validation set.
    Returns MAE on the inspiratory phase.
    """
    model.eval()
    total_mae = 0.0
    total_valid_steps = 0

    l1_fn = nn.L1Loss(reduction="none")

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            u_out = inputs[:, :, u_out_idx]
            mask = 1 - u_out

            # Forward pass (only care about final prediction)
            final_pred, _ = model(inputs)
            final_pred = final_pred.squeeze(-1)

            # Calculate absolute error
            abs_err = l1_fn(final_pred, targets)

            # Mask and sum
            masked_err = abs_err * mask

            total_mae += masked_err.sum().item()
            total_valid_steps += mask.sum().item()

    return total_mae / (total_valid_steps + 1e-8)


def train_model(train_loader, val_loader, config=Config):
    """
    Main training loop.
    """
    # Set seeds for reproducibility
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"Training on device: {device}")

    # Initialize Model
    model = VentilatorModel(config).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR_MAX, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    # OneCycleLR requires steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR_MAX,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup for first 10%
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # Loss Function
    loss_fn = MaskedL1Loss(aux_weight=config.AUX_WEIGHT)

    # Identify u_out index
    try:
        u_out_idx = config.INPUT_FEATURES.index("u_out")
    except ValueError:
        raise ValueError(
            "'u_out' not found in Config.INPUT_FEATURES. Cannot perform masked training."
        )

    best_mae = float("inf")

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, u_out_idx
        )

        val_mae = validate(model, val_loader, device, u_out_idx)

        # Checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(
                f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss} - Val MAE: {val_mae} (New Best)"
            )
        else:
            print(
                f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss} - Val MAE: {val_mae}"
            )

    print(f"Training complete. Best Validation MAE: {best_mae}")
    return best_mae


def predict_and_submit(test_loader, config=Config):
    """
    Generates predictions using the best model and saves submission.csv.
    """
    device = torch.device(config.DEVICE)

    # Initialize model
    model = VentilatorModel(config).to(device)

    # Load best weights
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}")

    print(f"Loading model from {config.MODEL_PATH}...")
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            final_pred, _ = model(inputs)

            # Flatten: (batch, seq_len, 1) -> (batch * seq_len)
            preds_flat = final_pred.view(-1).cpu().numpy()
            predictions.append(preds_flat)

    # Concatenate all predictions
    all_predictions = np.concatenate(predictions)

    # Load IDs
    test_ids = load_test_ids(config)

    # Sanity check
    if len(all_predictions) != len(test_ids):
        print(
            f"Warning: Prediction count {len(all_predictions)} != ID count {len(test_ids)}"
        )
        # Truncate or pad if necessary, though this implies a bug upstream
        min_len = min(len(all_predictions), len(test_ids))
        all_predictions = all_predictions[:min_len]
        test_ids = test_ids[:min_len]

    # Create DataFrame
    submission = pd.DataFrame({"id": test_ids, "pressure": all_predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
