import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders

# -------------------------------------------------------------------------
# Layers and Model Definition
# -------------------------------------------------------------------------


class GaussianSmearing(nn.Module):
    """
    Expands distances using a set of Gaussian Radial Basis Functions (RBFs).
    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        # Calculate width (coeff) based on spacing
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [E, 1] or [E]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AmplifiedCGCNNConv(MessagePassing):
    """
    Custom CGCNN Layer with Amplified Residual Connection.
    Update Rule: h_new = Softplus(2 * h_old + Message(h_old, neighbors, edges))
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # CGCNN message calculation: z_ij = cat(x_i, x_j, e_ij)
        # Input dimension = 2 * node_dim + edge_dim
        input_dim = 2 * node_dim + edge_dim
        self.lin = nn.Linear(input_dim, 2 * node_dim)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(node_dim)
        self.residual_scale = Config.RESIDUAL_SCALE

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim]
        # edge_index: [2, E]
        # edge_attr: [E, edge_dim]

        x_old = x

        # Propagate messages
        # This calls message(), aggregates, and returns the result
        message_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply BatchNorm to the aggregated message signal
        message_out = self.bn(message_out)

        # Amplified Residual Update
        # We use Softplus as the activation function to ensure smooth, positive outputs
        # The factor '2' (residual_scale) enforces the strong identity preservation bias
        out = self.softplus(self.residual_scale * x_old + message_out)

        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i, x_j: [E, node_dim]
        # edge_attr: [E, edge_dim]

        # Concatenate source, target, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Linear transformation
        z = self.lin(z)

        # Split into filter and core parts (CGCNN style gating)
        gate, trans = z.chunk(2, dim=-1)
        gate = self.sigmoid(gate)
        trans = self.softplus(trans)

        return gate * trans


class HASCNet(nn.Module):
    """
    Hybrid Amplified-Structural and Compositional Network.
    Combines a GNN backbone with a parallel MLP for global features.
    """

    def __init__(self):
        super().__init__()

        # --- Structural Stream (GNN) ---
        # Embedding for atomic numbers
        self.embedding = nn.Embedding(Config.ATOM_INPUT_DIM, Config.HIDDEN_DIM)

        # RBF Distance Expansion and Projection
        self.rbf = GaussianSmearing(
            start=0.0, stop=Config.CUTOFF_RADIUS, n_gaussians=Config.N_RBF
        )
        self.edge_proj = nn.Linear(Config.N_RBF, Config.HIDDEN_DIM)

        # GNN Layers
        self.convs = nn.ModuleList()
        for _ in range(Config.NUM_LAYERS):
            self.convs.append(AmplifiedCGCNNConv(Config.HIDDEN_DIM, Config.HIDDEN_DIM))

        # --- Compositional Stream (Global MLP) ---
        # Processes lattice parameters and composition fractions
        self.global_mlp = nn.Sequential(
            nn.Linear(Config.GLOBAL_FEATURE_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
        )

        # --- Fusion & Readout ---
        # Concatenate pooled graph features (HIDDEN_DIM) + global features (HIDDEN_DIM)
        fusion_dim = 2 * Config.HIDDEN_DIM

        # Decoupled Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Decoupled Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, global_x, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.global_x,
            data.batch,
        )

        # 1. Structural Stream
        h = self.embedding(x)  # [N_atoms, HIDDEN_DIM]

        # Process edges
        edge_attr = self.rbf(edge_attr)  # [E, N_RBF]
        edge_attr = self.edge_proj(edge_attr)  # [E, HIDDEN_DIM]

        # Message Passing
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)

        # Global Pooling (Mean)
        h_pool = global_mean_pool(h, batch)  # [Batch, HIDDEN_DIM]

        # 2. Compositional Stream
        h_global = self.global_mlp(global_x)  # [Batch, HIDDEN_DIM]

        # 3. Fusion
        h_fused = torch.cat([h_pool, h_global], dim=1)  # [Batch, 2*HIDDEN_DIM]

        # 4. Readout
        out_formation = self.head_formation(h_fused)
        out_bandgap = self.head_bandgap(h_fused)

        # Concatenate outputs: [Batch, 2]
        return torch.cat([out_formation, out_bandgap], dim=1)


# -------------------------------------------------------------------------
# Training and Evaluation Functions
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def run_training():
    """
    Main training loop for HASC-Net.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, _, _ = get_dataloaders()

    # 2. Initialize Model
    model = HASCNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")


def generate_submission():
    """
    Generates predictions for the test set using the best trained model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Prepare Data (Test set)
    # We need the target_scaler to inverse transform predictions
    _, _, test_loader, target_scaler = get_dataloaders()

    # 2. Load Model
    model = HASCNet().to(device)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Run training first."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Generating predictions...")
    ids = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)

            # Inverse transform predictions to original scale
            out_inv = target_scaler.inverse_transform(out)

            ids.extend(data.material_id)
            predictions.append(out_inv.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # 3. Save Submission
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
