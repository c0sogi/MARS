import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library import config, dataset
from library.model_components import (
    HybridInputStem,
    DenseDilatedBackbone,
    FeedbackStem,
    InteractionModule,
)

# =============================================================================
# MODEL DEFINITION
# =============================================================================


class StaticEncoder(nn.Module):
    """
    Encodes static features into latent representation Z.
    Wraps HybridInputStem and DenseDilatedBackbone.
    """

    def __init__(self):
        super().__init__()
        self.input_stem = HybridInputStem(
            in_channels=18, context_channels=config.GROWTH_RATE
        )
        self.backbone = DenseDilatedBackbone(
            in_channels=18 + config.GROWTH_RATE,
            growth_rate=config.GROWTH_RATE,
            dilations=config.DILATIONS,
            latent_dim=config.LATENT_DIM,
            dropout=config.DROPOUT,
        )

    def forward(self, x):
        # x: (N, 18, L)
        x_hybrid = self.input_stem(x)
        z = self.backbone(x_hybrid)
        return z


class RecurrentDecoder(nn.Module):
    """
    Decodes latent Z and feedback Y_prev into predictions.
    Wraps FeedbackStem and InteractionModule.
    """

    def __init__(self):
        super().__init__()
        self.feedback_stem = FeedbackStem(
            in_channels=5, hidden_dim=32, growth_rate=12, layers=4
        )
        self.interaction = InteractionModule(
            dim_z=config.LATENT_DIM,
            dim_fb=config.FEEDBACK_DIM,
            rnn_hidden=config.RNN_HIDDEN,
            num_targets=5,
        )

    def forward(self, z, y_prev, partner_indices):
        # z: (N, 64, L)
        # y_prev: (N, 5, L)
        # partner_indices: (N, L)
        e_fb = self.feedback_stem(y_prev)
        y_pred = self.interaction(z, e_fb, partner_indices)
        return y_pred


class HIGFDN(nn.Module):
    """
    Hybrid-Input Global-Feedback Dense Network.
    Wrapper for StaticEncoder and RecurrentDecoder.
    """

    def __init__(self):
        super().__init__()
        self.static_encoder = StaticEncoder()
        self.recurrent_decoder = RecurrentDecoder()

    def forward(self, x, partner_indices, y_prev=None):
        z = self.static_encoder(x)

        if y_prev is None:
            batch_size, _, length = x.shape
            y_prev = torch.zeros(
                (batch_size, 5, length), device=x.device, dtype=x.dtype
            )

        return self.recurrent_decoder(z, y_prev, partner_indices)


# =============================================================================
# LOSS FUNCTION
# =============================================================================


def masked_mcrmse_loss(preds, targets):
    """
    Calculates MCRMSE only on scored columns and scored positions.

    Args:
        preds: (N, 5, 107)
        targets: (N, 5, 107)

    Returns:
        loss: scalar
    """
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_cols = [0, 1, 3]

    # Scored length: 68
    scored_len = config.SCORED_LEN

    # Slice data
    preds_masked = preds[:, scored_cols, :scored_len]
    targets_masked = targets[:, scored_cols, :scored_len]

    # Calculate MSE per element
    mse = (preds_masked - targets_masked) ** 2

    # Average over Batch and Sequence (N * L) -> (C,)
    # Note: MCRMSE definition is mean of RMSEs of columns.
    # RMSE_j = sqrt(mean((y_ij - yhat_ij)^2))
    rmse_per_col = torch.sqrt(torch.mean(mse, dim=(0, 2)))

    # Mean over columns
    return torch.mean(rmse_per_col)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================


def validate(model, dataloader, device):
    model.eval()
    total_sse = torch.zeros(3, device=device)
    total_count = 0

    scored_cols = [0, 1, 3]
    scored_len = config.SCORED_LEN

    with torch.no_grad():
        for x, p, y in dataloader:
            x, p, y = x.to(device), p.to(device), y.to(device)

            # Inference: 2 Passes
            z = model.static_encoder(x)

            # Pass 1
            batch_size, _, length = x.shape
            y_prev = torch.zeros((batch_size, 5, length), device=device, dtype=x.dtype)
            y_pred_1 = model.recurrent_decoder(z, y_prev, p)

            # Pass 2
            y_pred_2 = model.recurrent_decoder(z, y_pred_1, p)

            # Calculate SSE for scored region
            preds_masked = y_pred_2[:, scored_cols, :scored_len]
            targets_masked = y[:, scored_cols, :scored_len]

            sse = torch.sum((preds_masked - targets_masked) ** 2, dim=(0, 2))
            total_sse += sse
            total_count += batch_size * scored_len

    # Global RMSE
    rmse_per_col = torch.sqrt(total_sse / total_count)
    mcrmse = torch.mean(rmse_per_col).item()

    return mcrmse


def train_model():
    config.set_seed()
    device = config.get_device()
    print(f"Using device: {device}")

    # Load Data
    train_dataset = dataset.RNADataset(split="train", debug=config.DEBUG)
    val_dataset = dataset.RNADataset(split="val", debug=config.DEBUG)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = HIGFDN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for i, (x, p, y) in enumerate(train_loader):
            x, p, y = x.to(device), p.to(device), y.to(device)

            optimizer.zero_grad()

            # 1. Static Encoding
            z = model.static_encoder(x)

            # 2. Recycling Pass 1
            batch_size, _, length = x.shape
            y_prev_0 = torch.zeros(
                (batch_size, 5, length), device=device, dtype=x.dtype
            )
            y_pred_1 = model.recurrent_decoder(z, y_prev_0, p)

            # 3. Recycling Pass 2
            # Detach gradients from Pass 1 output to stop gradient flow through feedback
            y_prev_1 = y_pred_1.detach()
            y_pred_2 = model.recurrent_decoder(z, y_prev_1, p)

            # 4. Loss Calculation
            # L = MCRMSE(Y2) + 0.5 * MCRMSE(Y1)
            loss1 = masked_mcrmse_loss(y_pred_1, y)
            loss2 = masked_mcrmse_loss(y_pred_2, y)
            loss = loss2 + 0.5 * loss1

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        val_mcrmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        scheduler.step(val_mcrmse)

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  New best model saved! ({val_mcrmse:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.6f}")


def generate_submission():
    print("Generating submission...")
    device = config.get_device()

    # Load Test Data
    test_dataset = dataset.RNADataset(split="test", debug=config.DEBUG)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Load Model
    model = HIGFDN().to(device)
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Model weights not found. Using random initialization.")

    model.eval()

    # Store predictions
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for x, p, _ in test_loader:
            x, p = x.to(device), p.to(device)

            # Inference: 2 Passes
            z = model.static_encoder(x)

            # Pass 1
            batch_size, _, length = x.shape
            y_prev = torch.zeros((batch_size, 5, length), device=device, dtype=x.dtype)
            y_pred_1 = model.recurrent_decoder(z, y_prev, p)

            # Pass 2
            y_pred_2 = model.recurrent_decoder(z, y_pred_1, p)

            # Move to CPU
            preds_np = y_pred_2.cpu().numpy()  # (B, 5, 107)

            # Transpose to (B, 107, 5) for ease of processing
            preds_np = preds_np.transpose(0, 2, 1)

            all_preds.append(preds_np)

            # Get IDs for this batch
            # Note: DataLoader shuffles=False, so order is preserved.
            # But we need IDs. dataset.RNADataset stores them.
            # We need to access them via the dataset index, but DataLoader batches make this tricky.
            # Easier way: The dataset has 'ids' list. We can iterate it later or return IDs in __getitem__.
            # Since __getitem__ doesn't return IDs, we rely on the order.

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, 107, 5)

    # Get IDs from dataset
    test_ids = test_dataset.ids

    # Prepare Submission Data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for j, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[j])

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def main():
    train_model()
    generate_submission()


if __name__ == "__main__":
    main()
