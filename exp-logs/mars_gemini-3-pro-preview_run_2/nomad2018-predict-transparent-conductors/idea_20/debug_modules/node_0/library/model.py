import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import os
import sys

# Import from library
from library.data import get_dataloaders
from library.utils import set_seed, compute_rmsle, StandardScaler

# ------------------------------------------------------------------------------
# Model Components
# ------------------------------------------------------------------------------


class CGConv(MessagePassing):
    """
    Gated Graph Convolution Layer.
    """

    def __init__(self, node_dim, edge_dim):
        super(CGConv, self).__init__(aggr="add")
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # We concatenate source node, target node, and edge features
        # Input dim: node_dim * 2 + edge_dim
        # Output dim: node_dim * 2 (one for filter, one for gate)
        self.linear = nn.Linear(node_dim * 2 + edge_dim, node_dim * 2)
        self.bn = nn.BatchNorm1d(node_dim * 2)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim]
        # edge_index: [2, E]
        # edge_attr: [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: [E, node_dim] (target)
        # x_j: [E, node_dim] (source)
        # edge_attr: [E, edge_dim]

        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        z = self.linear(z)
        z = self.bn(z)

        filter_part, gate_part = z.chunk(2, dim=1)
        return F.softplus(filter_part) * F.sigmoid(gate_part)

    def update(self, aggr_out):
        return aggr_out


class AdaptiveResidualBlock(nn.Module):
    """
    Residual Block with Learnable Scalar Residual.
    h_{l+1} = Softplus(BatchNorm(CGConv(h_l) + (1 + epsilon) * h_l))
    """

    def __init__(self, node_dim, edge_dim, dropout=0.1):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv = CGConv(node_dim, edge_dim)
        self.bn = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable scalar epsilon, initialized to 0.0
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_attr):
        # Convolution
        h_conv = self.conv(x, edge_index, edge_attr)

        # Dropout on convolution output
        h_conv = self.dropout(h_conv)

        # Adaptive Identity path
        h_res = (1.0 + self.epsilon) * x

        # Residual connection
        out = h_conv + h_res

        # Batch Norm and Activation
        out = self.bn(out)
        out = F.softplus(out)

        return out


class AICGN(nn.Module):
    """
    Adaptive-Identity Crystal Graph Network.
    """

    def __init__(
        self,
        node_input_dim=100,
        edge_input_dim=60,
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
    ):
        super(AICGN, self).__init__()

        # Node embedding (Atomic numbers -> hidden_dim)
        self.node_embedding = nn.Embedding(node_input_dim, hidden_dim)

        # Edge embedding (RBF -> hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)

        # Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                AdaptiveResidualBlock(hidden_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Decoupled Heads
        self.head_formation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.head_bandgap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Embeddings
        h = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        # Interaction Layers
        for block in self.blocks:
            h = block(h, edge_index, e)

        # Global Pooling (Mean)
        h_pool = global_mean_pool(h, batch)

        # Prediction Heads
        out_formation = self.head_formation(h_pool)
        out_bandgap = self.head_bandgap(h_pool)

        return out_formation, out_bandgap


# ------------------------------------------------------------------------------
# Training and Evaluation Logic
# ------------------------------------------------------------------------------


def train_model(
    model, train_loader, val_loader, device, epochs=100, lr=1e-3, weight_decay=1e-4
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Target Scaler
    scaler = StandardScaler(device)

    # Collect all targets to fit scaler
    all_targets = []
    for data in train_loader:
        all_targets.append(data.y)
    all_targets = torch.cat(all_targets, dim=0)
    scaler.fit(all_targets)

    best_val_loss = float("inf")
    patience = 15
    counter = 0
    best_model_state = None

    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            pred_form, pred_band = model(data)
            pred = torch.cat([pred_form, pred_band], dim=1)

            # Transform targets
            target_norm = scaler.transform(data.y)

            loss = criterion(pred, target_norm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_form, pred_band = model(data)
                pred_norm = torch.cat([pred_form, pred_band], dim=1)

                # Loss on normalized data
                target_norm = scaler.transform(data.y)
                loss = criterion(pred_norm, target_norm)
                val_loss += loss.item() * data.num_graphs

                # Inverse transform for RMSLE calculation
                pred_orig = scaler.inverse_transform(pred_norm)
                val_preds.append(pred_orig)
                val_targets.append(data.y)

        val_loss /= len(val_loader.dataset)

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # Calculate RMSLE on original scale
        # Clamp negative predictions to 0 for RMSLE
        val_rmsle = compute_rmsle(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSLE: {val_rmsle:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return best_model_state, scaler


def generate_predictions(model, test_loader, scaler, device, output_path):
    model.eval()
    ids = []
    preds_form = []
    preds_band = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            p_form, p_band = model(data)
            pred_norm = torch.cat([p_form, p_band], dim=1)
            pred_orig = scaler.inverse_transform(pred_norm)

            # Ensure non-negative
            pred_orig = torch.clamp(pred_orig, min=0.0)

            ids.extend(data.id.cpu().numpy())
            preds_form.extend(pred_orig[:, 0].cpu().numpy())
            preds_band.extend(pred_orig[:, 1].cpu().numpy())

    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds_form,
            "bandgap_energy_ev": preds_band,
        }
    )

    # Sort by ID just in case
    df = df.sort_values("id")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------


def run_pipeline():
    # Configuration
    BATCH_SIZE = 48
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    DROPOUT = 0.1
    EPOCHS = 150
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 42

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    # The library handles caching in ./working/idea_20/ if we pass load_cached_data=True
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True
    )

    # Model
    model = AICGN(
        node_input_dim=100,  # Atomic numbers up to ~100
        edge_input_dim=60,  # Gaussian smearing bins
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    # Training
    best_state, scaler = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Load best model
    model.load_state_dict(best_state)

    # Submission
    generate_predictions(
        model, test_loader, scaler, device, "./submission/submission.csv"
    )


if __name__ == "__main__":
    run_pipeline()
