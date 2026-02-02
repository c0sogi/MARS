import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import Config
from library.layers import DilatedResidualStem, StabilizedGLUInteraction
from library.utils import mcrmse_metric


class HighCapacityBiGRU(nn.Module):
    """
    High-Capacity BiGRU with Dilated Motif-Encoding Stem.

    Architecture:
    1. Dilated Residual Stem: Encodes local motifs (approx 15nt context).
    2. 4-Layer Backbone:
       - BiGRU (384 hidden per dir -> 768 total)
       - Stabilized GLU-Decoupled Interaction (Global folding constraints)
       - Dropout
    3. Output Head: Linear projection to 5 targets.
    """

    def __init__(self):
        super().__init__()

        # 1. Dilated Residual Stem
        self.stem = DilatedResidualStem(
            input_dim=Config.INPUT_DIM,
            filters=Config.STEM_FILTERS,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dilations=Config.STEM_DILATIONS,
        )

        # 2. High-Capacity Backbone
        # We use a ModuleList to stack the blocks sequentially
        self.layers = nn.ModuleList()

        # Input dimension for the first RNN is the stem output (768)
        # Hidden dimension is 384 per direction -> 768 total
        rnn_input_dim = Config.STEM_FILTERS
        hidden_dim = Config.RNN_HIDDEN_DIM

        for _ in range(Config.RNN_LAYERS):
            layer_block = nn.ModuleDict(
                {
                    "gru": nn.GRU(
                        input_size=rnn_input_dim,
                        hidden_size=hidden_dim,
                        batch_first=True,
                        bidirectional=True,
                    ),
                    "interaction": StabilizedGLUInteraction(
                        hidden_dim=hidden_dim * 2  # BiGRU output is 2 * 384 = 768
                    ),
                    "dropout": nn.Dropout(Config.DROPOUT),
                }
            )
            self.layers.append(layer_block)

            # For subsequent layers, input is the output of the previous BiGRU (768)
            rnn_input_dim = hidden_dim * 2

        # 3. Output Head
        self.head = nn.Linear(rnn_input_dim, 5)

    def forward(self, x, adjacency):
        """
        Args:
            x (torch.Tensor): Input features (N, L, 14).
            adjacency (torch.Tensor): Pairing indices (N, L).

        Returns:
            torch.Tensor: Predictions (N, L, 5).
        """
        # Pass through Stem
        x = self.stem(x)  # (N, L, 768)

        # Pass through Backbone Layers
        for layer in self.layers:
            # BiGRU
            # Output shape: (N, L, 2*hidden_dim)
            x_rnn, _ = layer["gru"](x)

            # Interaction Module (GLU-Decoupled)
            # Injects structural information based on adjacency
            x_interact = layer["interaction"](x_rnn, adjacency)

            # Dropout
            x = layer["dropout"](x_interact)

        # Output Head
        out = self.head(x)  # (N, L, 5)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, adjacency, targets in loader:
        inputs = inputs.to(device)
        adjacency = adjacency.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, adjacency)

        # Compute MSE loss on all 5 targets
        loss = criterion(outputs, targets)

        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, adjacency, targets in loader:
            inputs = inputs.to(device)
            adjacency = adjacency.to(device)

            outputs = model(inputs, adjacency)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE using the library utility
    # This utility handles slicing to SEQ_SCORED and filtering columns
    score = mcrmse_metric(y_true, y_pred)

    return score


def train_model(model, train_loader, val_loader):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    device = Config.DEVICE
    model.to(device)

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Loss Function: MSE
    criterion = nn.MSELoss()

    best_score = float("inf")
    patience = 5
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.10f} | Val MCRMSE: {val_score:.10f}"
        )

        # Early Stopping & Model Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! Score: {best_score:.10f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_score:.10f}")


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, adjacency, _ in loader:  # Targets are ignored/dummy in test
            inputs = inputs.to(device)
            adjacency = adjacency.to(device)

            outputs = model(inputs, adjacency)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
