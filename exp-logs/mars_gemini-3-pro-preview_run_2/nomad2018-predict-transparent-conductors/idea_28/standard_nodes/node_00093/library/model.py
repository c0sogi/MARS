import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import copy
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config
from library.utils import get_scaler, compute_metric
from library.data import get_dataloaders

# -------------------------------------------------------------------------
# Model Components
# -------------------------------------------------------------------------


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of radial basis functions.
    """

    def __init__(self, start=0.0, stop=Config.CUTOFF_RADIUS, n_bins=Config.RBF_BINS):
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_bins = n_bins
        offset = torch.linspace(start, stop, n_bins)
        # Width of RBFs
        self.gamma = -0.5 / ((stop - start) / n_bins) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges]
        # output: [num_edges, n_bins]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.gamma * torch.pow(dist, 2))


class ReceiverAwareConv(MessagePassing):
    """
    Receiver-Aware Gated Convolution.
    Message is computed from concatenation of Target, Source, and Edge features.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="sum")
        self.in_node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Input to linear layers is h_i (target) + h_j (source) + e_ij
        input_dim = 2 * node_dim + edge_dim

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(input_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: Target node features [num_edges, node_dim]
        # x_j: Source node features [num_edges, node_dim]
        # edge_attr: Edge features [num_edges, edge_dim]

        # Concatenate features: z_ij = [h_i || h_j || e_ij]
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gating mechanism
        # m_ij = Softplus(Linear1(z_ij)) * Sigmoid(Linear2(z_ij))
        gate = torch.sigmoid(self.lin2(z_ij))
        content = F.softplus(self.lin1(z_ij))

        return content * gate


class AdaptiveResidualBlock(nn.Module):
    """
    Sum-Normalized Adaptive Residual Block.
    Applies BatchNorm to the sum of the aggregated message and the residual.
    Includes a learnable epsilon for the identity path.
    """

    def __init__(self, node_dim, edge_dim, dropout):
        super().__init__()
        self.conv = ReceiverAwareConv(node_dim, edge_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Message passing aggregation
        m_agg = self.conv(x, edge_index, edge_attr)

        # Adaptive residual connection
        # h_l+1 = Softplus(BatchNorm(Agg(m_ij) + (1 + epsilon) * h_l))
        residual = (1 + self.epsilon) * x
        out = m_agg + residual

        out = self.bn(out)
        out = F.softplus(out)
        out = self.dropout(out)

        return out


class SRACGN(nn.Module):
    """
    Stabilized Receiver-Aware Crystal Graph Network.
    """

    def __init__(self):
        super().__init__()

        # Embeddings
        # Atomic numbers go up to ~100, 118 is safe upper bound
        self.node_embedding = nn.Embedding(118, Config.NODE_EMBED_DIM)
        self.rbf = GaussianRBF(0.0, Config.CUTOFF_RADIUS, Config.RBF_BINS)

        # Shared Edge Projection
        self.edge_proj = nn.Linear(Config.RBF_BINS, Config.HIDDEN_DIM)

        # Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                AdaptiveResidualBlock(
                    node_dim=Config.HIDDEN_DIM,
                    edge_dim=Config.HIDDEN_DIM,  # Projected edge dim
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.NUM_INTERACTION_BLOCKS)
            ]
        )

        # Readout Heads
        # Formation Energy Head
        self.head_formation = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 64),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(64, 1),
        )

        # Bandgap Energy Head
        self.head_bandgap = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 64),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial node features
        h = self.node_embedding(x)  # [num_nodes, hidden_dim]

        # Edge features expansion and projection
        # edge_attr comes in as distances [num_edges, 1]
        rbf_feat = self.rbf(edge_attr.squeeze())  # [num_edges, rbf_bins]
        e = self.edge_proj(rbf_feat)  # [num_edges, hidden_dim]
        e = F.silu(e)  # Activation on edge projection

        # Interaction Blocks
        for block in self.blocks:
            h = block(h, edge_index, e)

        # Global Pooling
        h_pool = global_mean_pool(h, batch)  # [batch_size, hidden_dim]

        # Prediction Heads
        out_formation = self.head_formation(h_pool)
        out_bandgap = self.head_bandgap(h_pool)

        return torch.cat([out_formation, out_bandgap], dim=1)


# -------------------------------------------------------------------------
# Training and Evaluation Functions
# -------------------------------------------------------------------------


def train_model():
    """
    Trains the S-RA-CGN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Get Scaler (fit on training data)
    # We need to reconstruct the full training targets array to fit the scaler
    # This is a bit inefficient with the loader, but we can do it once.
    # Alternatively, we can assume the scaler handles its own loading.
    # For robustness, let's just collect targets from the dataset.
    all_train_targets = []
    for data in train_loader.dataset:
        all_train_targets.append(data.y.numpy())
    all_train_targets = np.concatenate(all_train_targets, axis=0)

    scaler = get_scaler(all_train_targets, load_cached_data=True)

    # Initialize Model
    model = SRACGN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    early_stopping_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)

            # Scale targets
            targets = batch.y
            scaled_targets = torch.tensor(
                scaler.transform(targets.cpu().numpy()), dtype=torch.float32
            ).to(device)

            loss = criterion(outputs, scaled_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)

                # Calculate loss on scaled values for scheduler/stopping
                targets = batch.y
                scaled_targets = torch.tensor(
                    scaler.transform(targets.cpu().numpy()), dtype=torch.float32
                ).to(device)

                loss = criterion(outputs, scaled_targets)
                val_loss += loss.item() * batch.num_graphs

                # Inverse transform for metric calculation
                preds_np = scaler.inverse_transform(outputs.cpu().numpy())
                all_preds.append(preds_np)
                all_targets.append(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Calculate RMSLE
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_rmsle = compute_metric(all_targets, all_preds)

        # Scheduler step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSLE: {val_rmsle:.6f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            # print(f"  New best model saved to {Config.MODEL_CHECKPOINT_PATH}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")


def generate_submission():
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using device: {device}")

    # Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Scaler
    scaler = get_scaler(None, load_cached_data=True)  # Should load from cache

    # Load Model
    model = SRACGN().to(device)
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError("Model checkpoint not found. Run training first.")

    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Inverse transform
            preds_np = scaler.inverse_transform(outputs.cpu().numpy())

            all_preds.append(preds_np)
            all_ids.append(batch.id.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID to match sample submission format usually
    submission_df = submission_df.sort_values("id")

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    train_model()
    generate_submission()
