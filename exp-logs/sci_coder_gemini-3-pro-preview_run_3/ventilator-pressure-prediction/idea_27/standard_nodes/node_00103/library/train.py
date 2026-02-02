import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import prepare_data
from library.model import LGRHNet
from library.utils import seed_everything, get_device


class MaskedL1Loss(nn.Module):
    """
    Computes the Mean Absolute Error (L1 Loss) masked by the inspiratory phase.
    The metric is only calculated where u_out == 0 (inspiratory phase).
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (Batch, Seq_Len, 1)
            target: Ground truth (Batch, Seq_Len)
            u_out: Expiratory valve status (Batch, Seq_Len). 0=Inspiratory, 1=Expiratory.
        """
        # Ensure shapes match
        if pred.shape != target.shape:
            # If pred is (B, L, 1) and target is (B, L), squeeze pred
            pred = pred.squeeze(-1)

        # Calculate raw L1 loss
        loss = self.l1(pred, target)

        # Create mask: 1 for inspiratory (u_out=0), 0 for expiratory (u_out=1)
        mask = 1.0 - u_out

        # Apply mask
        masked_loss = loss * mask

        # Normalize by the number of valid inspiratory steps
        # Add epsilon to prevent division by zero
        score = masked_loss.sum() / (mask.sum() + 1e-8)

        return score


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: MaskedL1Loss,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (X, y, u_out) in enumerate(loader):
        X, y, u_out = X.to(device), y.to(device), u_out.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X)

        # Compute loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MaskedL1Loss,
    device: torch.device,
) -> float:
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y, u_out in loader:
            X, y, u_out = X.to(device), y.to(device), u_out.to(device)

            preds = model(X)
            loss = criterion(preds, y, u_out)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def predict_and_submit(
    model: nn.Module,
    loader: DataLoader,
    config: Config,
    device: torch.device,
):
    """
    Generates predictions for the test set and creates the submission file.
    """
    print("Generating predictions for test set...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device)
            preds = model(X)
            # preds shape: (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
            preds = preds.squeeze(-1)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches
    # Shape: (Num_Breaths, Seq_Len)
    predictions = np.concatenate(all_preds, axis=0)

    # Flatten to match submission format (1D array of all time steps)
    predictions_flat = predictions.flatten()

    # Load test IDs
    test_ids_path = os.path.join(config.cache_dir, "test_ids.npy")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(
            f"Test IDs not found at {test_ids_path}. Run prepare_data first."
        )

    test_ids = np.load(test_ids_path)

    # Ensure lengths match
    if len(test_ids) != len(predictions_flat):
        print(
            f"Warning: Length mismatch. IDs: {len(test_ids)}, Preds: {len(predictions_flat)}"
        )
        # In case of mismatch (e.g. drop_last in loader, though test loader shouldn't drop),
        # we strictly trust the IDs and truncate/pad if necessary, but here we raise error.
        raise ValueError("Prediction length does not match ID length.")

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "pressure": predictions_flat})

    # Save
    print(f"Saving submission to {config.submission_file}")
    submission_df.to_csv(config.submission_file, index=False)
    print("Submission saved successfully.")


def run_training(config: Config = Config()):
    """
    Main execution function for training and submission.
    """
    # 1. Setup
    seed_everything(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Preparation
    train_loader, val_loader, test_loader = prepare_data(config)

    # 3. Model Initialization
    # Get input dimension from first batch
    sample_X, _, _ = next(iter(train_loader))
    input_dim = sample_X.shape[2]

    model = LGRHNet(input_dim=input_dim, config=config)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    if config.scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience_scheduler,
            min_lr=config.min_lr,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
        )

    criterion = MaskedL1Loss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.max_grad_norm
        )

        val_loss = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        if config.scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  -> New best model saved! Score: {best_val_loss:.8f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{config.patience}"
            )

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))

    predict_and_submit(model, test_loader, config, device)
