import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_scatter import scatter
from library.data import get_train_val_loaders, get_test_loader

# Set fixed seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class AtomicEncoder(nn.Module):
    """
    Encodes local atomic features into a high-dimensional latent representation.
    Input: (N_atoms, 21) -> Output: (N_atoms, 512)
    """

    def __init__(self, input_dim=21, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GlobalEncoder(nn.Module):
    """
    Encodes global crystal features (lattice, stoichiometry, variance).
    Input: (Batch_Size, 22) -> Output: (Batch_Size, 256)
    """

    def __init__(self, input_dim=22, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CEADSModel(nn.Module):
    """
    Chemically-Explicit Anisotropic Deep Sets (CEADS) Model.
    Integrates atomic and global streams via Dual Pooling and Late Fusion.
    """

    def __init__(
        self,
        atomic_input_dim=21,
        global_input_dim=22,
        atomic_hidden=512,
        global_hidden=256,
        fusion_hidden=256,
        output_dim=2,
        dropout=0.1,
    ):
        super().__init__()

        self.atomic_encoder = AtomicEncoder(atomic_input_dim, atomic_hidden, dropout)
        self.global_encoder = GlobalEncoder(global_input_dim, global_hidden, dropout)

        # Dual Pooling (Mean + Max) doubles the atomic representation size
        fusion_input_dim = (2 * atomic_hidden) + global_hidden

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, output_dim),
        )

    def forward(self, atomic_feats, batch_indices, global_feats):
        """
        Args:
            atomic_feats: (N_total_atoms, 21)
            batch_indices: (N_total_atoms,) mapping atoms to batch index
            global_feats: (Batch_Size, 22)
        """
        # 1. Atomic Stream
        atom_emb = self.atomic_encoder(atomic_feats)  # (N_atoms, atomic_hidden)

        # 2. Aggregation (Dual Pooling)
        batch_size = global_feats.size(0)
        # Scatter Mean
        pooled_mean = scatter(
            atom_emb, batch_indices, dim=0, dim_size=batch_size, reduce="mean"
        )
        # Scatter Max
        pooled_max = scatter(
            atom_emb, batch_indices, dim=0, dim_size=batch_size, reduce="max"
        )

        # 3. Global Stream
        glob_emb = self.global_encoder(global_feats)  # (Batch_Size, global_hidden)

        # 4. Late Fusion
        combined = torch.cat([pooled_mean, pooled_max, glob_emb], dim=1)

        # 5. Prediction
        out = self.fusion_head(combined)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch from SparseCollate
        atomic_feats, batch_indices, global_feats, targets, _ = batch

        # Move to device
        atomic_feats = atomic_feats.to(device)
        batch_indices = batch_indices.to(device)
        global_feats = global_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(atomic_feats, batch_indices, global_feats)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * global_feats.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            atomic_feats, batch_indices, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            outputs = model(atomic_feats, batch_indices, global_feats)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * global_feats.size(0)

    return running_loss / len(loader.dataset)


def generate_submission(model, device, output_path="./submission/submission.csv"):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    test_loader = get_test_loader(batch_size=64, num_workers=2)

    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_feats, batch_indices, global_feats, _, ids = batch

            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats = global_feats.to(device)

            # Predict (log space)
            outputs = model(atomic_feats, batch_indices, global_feats)

            # Inverse transform: exp(x) - 1
            preds = torch.expm1(outputs).cpu().numpy()
            ids = ids.numpy()

            for i in range(len(ids)):
                results.append(
                    {
                        "id": ids[i],
                        "formation_energy_ev_natom": preds[i, 0],
                        "bandgap_energy_ev": preds[i, 1],
                    }
                )

    # Create DataFrame and save
    df = pd.DataFrame(results)
    # Ensure correct column order
    df = df[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 150
    PATIENCE = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # Data Loaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=BATCH_SIZE, num_workers=2
    )

    # Model Initialization
    model = CEADSModel(
        atomic_input_dim=21,
        global_input_dim=22,
        atomic_hidden=512,
        global_hidden=256,
        fusion_hidden=256,
        output_dim=2,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state (in memory or file if needed, here we just keep it in memory for submission)
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for submission
    if "best_model_state" in locals():
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val Loss: {best_val_loss:.6f}")

    # Generate Submission
    generate_submission(model, DEVICE)


if __name__ == "__main__":
    main()
