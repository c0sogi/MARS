import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config
from library.data import get_dataloaders
from library.utils import set_seed

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class GaussianRBF(nn.Module):
    def __init__(self, start=0.0, end=5.0, num_bins=60, gamma=10.0):
        super().__init__()
        self.centers = nn.Parameter(
            torch.linspace(start, end, num_bins), requires_grad=False
        )
        self.gamma = gamma

    def forward(self, dist):
        # dist: (E, 1) or (E,)
        if dist.dim() == 2:
            dist = dist.squeeze(1)
        # (E, 1) - (bins,) -> (E, bins) via broadcasting
        return torch.exp(-self.gamma * (dist.unsqueeze(1) - self.centers) ** 2)


class ReceiverAwareBlock(MessagePassing):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Message function: Gated MLP on [h_i, h_j, e_ij]
        # Input dim: hidden_dim (i) + hidden_dim (j) + hidden_dim (e) = 3 * hidden_dim
        # Project to 2 * hidden_dim for GLU-like gating
        self.message_mlp = nn.Linear(3 * hidden_dim, 2 * hidden_dim)

        # Learnable scalar residual
        self.epsilon = nn.Parameter(torch.zeros(1))

        # BatchNorm + Activation for update
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.Softplus()

    def forward(self, x, edge_index, edge_attr):
        # x: (N, hidden_dim)
        # edge_index: (2, E)
        # edge_attr: (E, hidden_dim) - already projected

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Residual connection with learnable scalar
        out = out + (1.0 + self.epsilon) * x

        # BatchNorm + Activation
        out = self.bn(out)
        out = self.act(out)

        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i: target nodes (E, hidden_dim)
        # x_j: source nodes (E, hidden_dim)
        # edge_attr: (E, hidden_dim)

        # Concatenate
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)  # (E, 3*hidden_dim)

        # Gated MLP projection
        gate_input = self.message_mlp(z)  # (E, 2*hidden_dim)

        # Split into value and gate
        val, gate = torch.split(gate_input, self.hidden_dim, dim=-1)

        # Apply gating
        msg = val * torch.sigmoid(gate)

        return F.dropout(msg, p=self.dropout, training=self.training)


class SpeciesWeightedReadout(nn.Module):
    def __init__(self, max_atomic_num, hidden_dim):
        super().__init__()
        # Learnable weights for each atomic number
        # Initialize to 0 (softplus(0) approx 0.69) to start near uniform
        self.species_weights = nn.Parameter(torch.zeros(max_atomic_num + 1))

    def forward(self, x, z, batch):
        # x: (N, hidden_dim)
        # z: (N,) atomic numbers
        # batch: (N,)

        # Get scalar weights for each atom based on atomic number
        w = self.species_weights[z]  # (N,)

        # Ensure positivity using Softplus
        w = F.softplus(w).unsqueeze(1)  # (N, 1)

        # Weight the node features
        x_weighted = x * w

        # Global Mean Pooling
        # This computes (1/N_g) * sum(w_i * x_i) for each graph g
        out = global_mean_pool(x_weighted, batch)

        return out


class SW_RA_CGN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.atom_embedding = nn.Embedding(config.MAX_ATOMIC_NUM + 1, config.HIDDEN_DIM)

        # RBF expansion
        self.rbf = GaussianRBF(
            start=0.0,
            end=config.CUTOFF,
            num_bins=config.RBF_BINS,
            gamma=config.RBF_GAMMA,
        )

        # Shared Linear Layer for edge projection
        self.edge_projection = nn.Linear(config.RBF_BINS, config.HIDDEN_DIM)

        # Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                ReceiverAwareBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # Species Weighted Readout
        self.readout = SpeciesWeightedReadout(config.MAX_ATOMIC_NUM, config.HIDDEN_DIM)

        # Decoupled Heads
        # Formation Energy Head
        self.head_formation = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 64),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1),
        )

        # Bandgap Energy Head
        self.head_bandgap = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 64),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch, z = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.x,
        )

        # Node Embeddings
        h = self.atom_embedding(x)  # (N, hidden_dim)

        # Edge Expansion and Projection (Shared)
        rbf_feat = self.rbf(edge_attr)  # (E, rbf_bins)
        e = self.edge_projection(rbf_feat)  # (E, hidden_dim)

        # Interaction Blocks
        for block in self.blocks:
            h = block(h, edge_index, e)

        # Readout
        h_graph = self.readout(h, z, batch)  # (B, hidden_dim)

        # Prediction Heads
        out_formation = self.head_formation(h_graph)
        out_bandgap = self.head_bandgap(h_graph)

        # Concatenate outputs
        return torch.cat([out_formation, out_bandgap], dim=1)


# -------------------------------------------------------------------------
# Training Logic
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        preds = model(batch)
        loss = criterion(preds, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        loss = criterion(preds, batch.y)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device, scaler):
    model.eval()
    all_preds = []
    all_ids = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)

        # Inverse transform to get original scale
        preds_np = preds.cpu().numpy()
        preds_original = scaler.inverse_transform(preds_np)

        all_preds.append(preds_original)
        all_ids.append(batch.id.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_ids)


def run_experiment():
    set_seed(Config.SEED)

    # 1. Data Loading
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=True
    )

    # 2. Model Initialization
    device = torch.device(Config.DEVICE)
    model = SW_RA_CGN(Config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    # 3. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print(f"  New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 4. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    predictions, ids = predict(model, test_loader, device, scaler)

    # 5. Submission
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure ID is integer
    submission_df["id"] = submission_df["id"].astype(int)
    # Sort by ID
    submission_df = submission_df.sort_values("id")

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    run_experiment()
