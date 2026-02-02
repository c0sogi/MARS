import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library import config
from library.dataset import get_dataloader

# Set seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)


class AtomicStream(nn.Module):
    """
    Processes per-atom features using a Wide MLP with Batch Normalization and Dropout.
    Aggregates features using Dual Pooling (Global Mean + Global Max).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(AtomicStream, self).__init__()

        # Wide MLP Encoder
        # We use a sequence of Linear -> BN -> ReLU -> Dropout blocks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),  # Projection to embedding space
        )

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, MaxAtoms, InputDim)
            mask: (Batch, MaxAtoms) - 1 for real atoms, 0 for padding
        Returns:
            embedding: (Batch, 2 * OutputDim) - Concatenated Mean and Max pooling
        """
        B, N, C = x.shape

        # Flatten to (B*N, C) to apply MLP point-wise
        x_flat = x.view(-1, C)

        # Apply Encoder
        # BatchNorm1d works on (N_samples, Features), so flattening is correct
        out_flat = self.encoder(x_flat)

        # Reshape back to (B, N, OutputDim)
        out = out_flat.view(B, N, -1)

        # Expand mask for broadcasting: (B, N, 1)
        mask_expanded = mask.unsqueeze(-1)

        # Mask the outputs (set padded values to 0)
        out_masked = out * mask_expanded

        # --- Aggregation ---

        # 1. Global Mean Pooling
        sum_out = torch.sum(out_masked, dim=1)  # (B, OutputDim)
        atom_counts = torch.sum(mask, dim=1, keepdim=True)  # (B, 1)
        atom_counts = torch.clamp(atom_counts, min=1.0)  # Avoid division by zero
        mean_pool = sum_out / atom_counts

        # 2. Global Max Pooling
        # Replace padded positions with a very small number before max
        # (1 - mask) gives 1 for padded positions
        out_for_max = out_masked + (1.0 - mask_expanded) * -1e9
        max_pool, _ = torch.max(out_for_max, dim=1)  # (B, OutputDim)

        # Concatenate poolings
        embedding = torch.cat([mean_pool, max_pool], dim=1)

        return embedding


class GlobalStream(nn.Module):
    """
    Processes global crystal features using a High-Capacity MLP.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GlobalStream, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x: (Batch, InputDim)
        return self.net(x)


class HCPDS(nn.Module):
    """
    Hybrid-Coordinate Potential Deep Sets Architecture.
    Integrates Atomic and Global streams via Late Fusion.
    """

    def __init__(self):
        super(HCPDS, self).__init__()

        # Dimensions from config
        atomic_in = config.ATOMIC_INPUT_DIM
        atomic_hidden = config.ATOMIC_HIDDEN_DIM

        # We project atomic features to a latent size before pooling.
        # Using 256 as the projection dimension.
        atomic_out_proj = 256

        global_in = config.GLOBAL_INPUT_DIM
        global_hidden = config.GLOBAL_HIDDEN_DIM
        global_out_proj = 128

        fusion_hidden = config.FUSION_HIDDEN_DIM
        dropout = config.DROPOUT_RATE

        # Streams
        self.atomic_stream = AtomicStream(
            atomic_in, atomic_hidden, atomic_out_proj, dropout
        )
        self.global_stream = GlobalStream(
            global_in, global_hidden, global_out_proj, dropout
        )

        # Fusion Input Dimension
        # Atomic stream returns (Mean + Max) -> 2 * atomic_out_proj
        # Global stream returns global_out_proj
        fusion_in = (2 * atomic_out_proj) + global_out_proj

        # Fusion Head (Regressor)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, 2),  # Targets: formation_energy, bandgap
        )

    def forward(self, atomic_features, global_features, mask):
        # Process Atomic Stream
        atomic_emb = self.atomic_stream(atomic_features, mask)

        # Process Global Stream
        global_emb = self.global_stream(global_features)

        # Late Fusion
        fusion_vec = torch.cat([atomic_emb, global_emb], dim=1)

        # Prediction
        output = self.fusion_head(fusion_vec)

        return output


def train_model():
    """
    Trains the HCPDS model with Early Stopping and LR Scheduling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DataLoaders (using cached data if available)
    train_loader = get_dataloader("train", batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("val", batch_size=config.BATCH_SIZE, shuffle=False)

    model = HCPDS().to(device)

    # Optimizer with Weight Decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            atomic_feat = batch["atomic_features"].to(device)
            global_feat = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(atomic_feat, global_feat, mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                atomic_feat = batch["atomic_features"].to(device)
                global_feat = batch["global_features"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["target"].to(device)

                outputs = model(atomic_feat, global_feat, mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

        val_loss /= len(val_loader.dataset)

        # Update Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


def generate_submission():
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using device: {device}")

    # Load Test Data
    test_loader = get_dataloader("test", batch_size=config.BATCH_SIZE, shuffle=False)

    # Load Model
    model = HCPDS().to(device)
    if not os.path.exists(config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {config.MODEL_CHECKPOINT}. Train model first."
        )

    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_feat = batch["atomic_features"].to(device)
            global_feat = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"].cpu().numpy()

            outputs = model(atomic_feat, global_feat, mask)

            # Inverse transform: exp(y) - 1 (since we trained on log1p)
            preds = torch.expm1(outputs).cpu().numpy()

            ids.extend(batch_ids)
            predictions.extend(preds)

    # Create DataFrame
    predictions = np.array(predictions)
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID
    submission_df.sort_values("id", inplace=True)

    # Save
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
