import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import global_max_pool
from library.config import Config
from library.dataset import get_dataloaders
from library.utils import set_seed, compute_rmsle, save_submission

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class AtomicBranch(nn.Module):
    """
    Deep Sets branch for processing atomic features and coordinates.
    Applies a shared MLP to each atom and aggregates via global max pooling.
    """

    def __init__(self):
        super().__init__()
        layers = []
        # Input dimension: Cartesian coords (3) + One-hot atom types
        input_dim = Config.ATOMIC_INPUT_DIM

        # Construct MLP layers
        # Sequence: Input -> Hidden_1 -> ... -> Hidden_N -> Latent
        layer_dims = Config.ATOMIC_HIDDEN_DIMS + [Config.ATOMIC_LATENT_DIM]

        for dim in layer_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, batch):
        """
        Args:
            x: Tensor of shape (num_atoms_in_batch, input_dim)
            batch: Tensor of shape (num_atoms_in_batch,) assigning atoms to graphs
        Returns:
            Tensor of shape (batch_size, atomic_latent_dim)
        """
        # Process each atom independently
        x_processed = self.mlp(x)

        # Aggregate features for each crystal (graph)
        # Max pooling is permutation invariant
        x_pooled = global_max_pool(x_processed, batch)
        return x_pooled


class LatticeBranch(nn.Module):
    """
    Branch for processing macroscopic lattice features.
    """

    def __init__(self):
        super().__init__()
        layers = []
        input_dim = Config.LATTICE_INPUT_DIM

        # Construct MLP layers
        layer_dims = Config.LATTICE_HIDDEN_DIMS + [Config.LATTICE_LATENT_DIM]

        for dim in layer_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, lattice_features):
        """
        Args:
            lattice_features: Tensor of shape (batch_size, 1, input_dim) or (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, lattice_latent_dim)
        """
        # Ensure correct shape (remove the middle dimension if present)
        if lattice_features.dim() == 3:
            lattice_features = lattice_features.squeeze(1)

        return self.mlp(lattice_features)


class LCDSModel(nn.Module):
    """
    Lattice-Conditioned Deep Sets Model.
    Fuses geometric fingerprints from AtomicBranch with lattice embeddings from LatticeBranch.
    """

    def __init__(self):
        super().__init__()
        self.atomic_branch = AtomicBranch()
        self.lattice_branch = LatticeBranch()

        # Fusion and Regression MLP
        layers = []
        # Input is concatenation of both branch outputs
        input_dim = Config.FUSION_INPUT_DIM

        for hidden_dim in Config.FUSION_HIDDEN_DIMS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT_RATE))
            input_dim = hidden_dim

        # Final prediction layer
        layers.append(nn.Linear(input_dim, Config.OUTPUT_DIM))

        self.regressor = nn.Sequential(*layers)

    def forward(self, data):
        """
        Args:
            data: PyG DataBatch object containing x, batch, lattice_features
        Returns:
            Tensor of shape (batch_size, 2) containing predicted log-energies
        """
        # Extract inputs
        x, batch = data.x, data.batch
        lattice_features = data.lattice_features

        # Get embeddings
        atomic_emb = self.atomic_branch(x, batch)
        lattice_emb = self.lattice_branch(lattice_features)

        # Concatenate
        combined = torch.cat([atomic_emb, lattice_emb], dim=1)

        # Predict
        out = self.regressor(combined)
        return out


# -------------------------------------------------------------------------
# Training and Evaluation Functions
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Targets are already log-transformed in the dataset
        targets = batch.y

        # Ensure shapes match (batch.y might be [B, 1, 2] or [B, 2])
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            targets = batch.y

            if targets.dim() == 3:
                targets = targets.squeeze(1)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * batch.num_graphs

            all_preds.append(outputs.cpu())
            # For RMSLE calculation, we need the original scale targets.
            # The dataset provides log(1+x). We can invert this or use the raw values if we had them.
            # However, compute_rmsle expects y_true in original scale and y_pred in log scale.
            # We can recover y_true original from y_log via expm1.
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all
    all_preds = torch.cat(all_preds, dim=0)
    all_targets_log = torch.cat(all_targets, dim=0)

    # Recover original scale for ground truth to compute RMSLE
    all_targets_orig = torch.expm1(all_targets_log).numpy()

    # Compute RMSLE
    rmsle_score = compute_rmsle(all_targets_orig, all_preds)

    return epoch_loss, rmsle_score


def generate_predictions(model, loader, device):
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            all_ids.extend(batch.id.cpu().numpy())
            all_preds.append(outputs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    return all_ids, all_preds


def run_experiment(load_cached_data=True):
    """
    Main function to run the training pipeline and generate submission.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    model = LCDSModel().to(device)

    # 4. Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    # MSE Loss on log-transformed targets
    criterion = nn.MSELoss()

    # 5. Training Loop
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_val_rmsle = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val RMSLE: {val_rmsle:.6f}"
        )

        # Early Stopping Check based on RMSLE
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation RMSLE: {best_val_rmsle:.6f}")

    # 6. Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 7. Generate Submission
    print("Generating predictions for test set...")
    test_ids, test_preds_log = generate_predictions(model, test_loader, device)

    save_submission(test_ids, test_preds_log, Config.SUBMISSION_PATH)
    print("Experiment completed successfully.")
