import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
import pandas as pd
import os
import math

from library.config import Config
from library.data import get_dataloaders
from library.utils import set_seed, TargetScaler

# Initialize Config
config = Config()


class GaussianRBF(nn.Module):
    def __init__(self, start=0.0, end=5.0, num_bins=60):
        super().__init__()
        self.start = start
        self.end = end
        self.num_bins = num_bins
        # Centers of the RBFs
        self.centers = nn.Parameter(
            torch.linspace(start, end, num_bins), requires_grad=False
        )
        # Width of the RBFs (gamma)
        self.gamma = nn.Parameter(
            torch.tensor([(2 * num_bins / (end - start)) ** 2]), requires_grad=False
        )

    def forward(self, d):
        # d: [num_edges, 1]
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


class ReceiverAwareInteractionBlock(MessagePassing):
    def __init__(self, hidden_channels, dropout):
        super().__init__(aggr="add")
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Input to message is [h_i || h_j || e_ij]
        # h_i, h_j are hidden_channels
        # e_ij is hidden_channels (projected RBF)
        input_dim = 3 * hidden_channels

        self.lin1 = nn.Linear(input_dim, hidden_channels)
        self.lin2 = nn.Linear(input_dim, hidden_channels)

        self.bn = nn.BatchNorm1d(hidden_channels)

        # Learnable epsilon for residual connection
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, hidden_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, hidden_channels] (Projected RBF)

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Stabilized Adaptive Residual Update
        # h^(l+1) = Softplus(BatchNorm(Agg(m_ij) + (1 + epsilon) * h^(l)))

        # Compute residual part
        residual = (1 + self.epsilon) * x

        # Sum before BN
        out = out + residual

        # BN
        out = self.bn(out)

        # Activation
        out = F.softplus(out)

        # Dropout
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i: Target node features [num_edges, hidden_channels]
        # x_j: Source node features [num_edges, hidden_channels]
        # edge_attr: Edge features [num_edges, hidden_channels]

        # Concatenate [h_i || h_j || e_ij]
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gated mechanism
        # m_ij = Softplus(Linear1(z)) * Sigmoid(Linear2(z))
        gate = torch.sigmoid(self.lin2(z_ij))
        content = F.softplus(self.lin1(z_ij))

        return content * gate


class IS_RA_CGN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config.hidden_channels
        self.num_rbf_bins = config.num_rbf_bins
        self.cutoff = config.cutoff_radius

        # 1. Embeddings
        # Atomic numbers up to 100 (safe upper bound)
        self.embedding = nn.Embedding(100, self.hidden_channels)

        # Edge Expansion
        self.rbf = GaussianRBF(start=0.0, end=self.cutoff, num_bins=self.num_rbf_bins)

        # Shared Linear Projection for edges
        self.edge_projection = nn.Linear(self.num_rbf_bins, self.hidden_channels)

        # 2. Interaction Backbone
        self.blocks = nn.ModuleList(
            [
                ReceiverAwareInteractionBlock(self.hidden_channels, config.dropout)
                for _ in range(config.num_interaction_blocks)
            ]
        )

        # 3. Prediction Heads (Input-Skip Readout)
        # Input to MLP is concatenation of initial and final pooled embeddings: 2 * hidden_channels
        input_dim = 2 * self.hidden_channels

        # Head 1: Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_channels // 2, 1),
        )

        # Head 2: Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_channels // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Node Embeddings (h^(0))
        h0 = self.embedding(x)

        # Edge Features
        # edge_attr is distances [num_edges, 1]
        rbf_out = self.rbf(edge_attr)  # [num_edges, num_rbf_bins]
        e_ij = self.edge_projection(rbf_out)  # [num_edges, hidden_channels]

        # Interaction Blocks
        h = h0
        for block in self.blocks:
            h = block(h, edge_index, e_ij)

        # Input-Skip Readout
        # Pool initial embeddings
        z_comp = global_mean_pool(h0, batch)

        # Pool final embeddings
        z_struct = global_mean_pool(h, batch)

        # Fusion
        z_final = torch.cat([z_struct, z_comp], dim=1)

        # Predictions
        out_formation = self.head_formation(z_final)
        out_bandgap = self.head_bandgap(z_final)

        return torch.cat([out_formation, out_bandgap], dim=1)


def train_model():
    set_seed(config.seed)

    # Data Loaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Model setup
    model = IS_RA_CGN(config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )
    criterion = nn.MSELoss()

    # Target Scaler
    scaler = TargetScaler()

    # Fit scaler on training data
    all_targets = []
    for data in train_loader:
        all_targets.append(data.y)
    all_targets = torch.cat(all_targets, dim=0)
    scaler.fit(all_targets)
    print(f"Targets Mean: {scaler.mean}, Std: {scaler.std}")

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data.to(config.device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Scale targets
            targets_scaled = scaler.transform(data.y)

            loss = criterion(outputs, targets_scaled)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae_formation = 0.0
        val_mae_bandgap = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(config.device)
                outputs = model(data)

                # Loss on scaled targets
                targets_scaled = scaler.transform(data.y)
                loss = criterion(outputs, targets_scaled)
                val_loss += loss.item() * data.num_graphs

                # Metrics on original scale
                preds_orig = scaler.inverse_transform(outputs)
                targets_orig = data.y

                mae = torch.abs(preds_orig - targets_orig)
                val_mae_formation += mae[:, 0].sum().item()
                val_mae_bandgap += mae[:, 1].sum().item()

        val_loss /= len(val_loader.dataset)
        val_mae_formation /= len(val_loader.dataset)
        val_mae_bandgap /= len(val_loader.dataset)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE Form: {val_mae_formation:.6f} | Val MAE Band: {val_mae_bandgap:.6f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(config.checkpoint_dir, "best_model.pth"),
            )
            scaler.save(os.path.join(config.cache_dir, "target_scaler.npz"))
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for submission
    print("Loading best model for submission...")
    model.load_state_dict(
        torch.load(os.path.join(config.checkpoint_dir, "best_model.pth"))
    )
    scaler.load(os.path.join(config.cache_dir, "target_scaler.npz"))
    model.eval()

    ids = []
    preds_formation = []
    preds_bandgap = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(config.device)
            outputs = model(data)
            preds_orig = scaler.inverse_transform(outputs)

            # Store IDs and predictions
            ids.extend(data.id)
            preds_formation.extend(preds_orig[:, 0].cpu().numpy())
            preds_bandgap.extend(preds_orig[:, 1].cpu().numpy())

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds_formation,
            "bandgap_energy_ev": preds_bandgap,
        }
    )

    submission_path = os.path.join(config.submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
