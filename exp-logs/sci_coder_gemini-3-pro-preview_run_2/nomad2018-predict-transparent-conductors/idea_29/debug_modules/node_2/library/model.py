import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
import pandas as pd
import os

from library import config
from library import utils
from library import data


class RBFExpansion(nn.Module):
    """
    Expands distances into a Gaussian Radial Basis Function (RBF) basis.
    """

    def __init__(self, vmin=0, vmax=8.0, bins=60):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        # Register buffer for centers and gamma so they are saved with the model but not trained
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.register_buffer("gamma", torch.tensor((bins / (vmax - vmin)) ** 2))

    def forward(self, x):
        """
        Args:
            x: Edge distances [num_edges, 1]
        Returns:
            RBF features [num_edges, bins]
        """
        return torch.exp(-self.gamma * (x - self.centers) ** 2)


class LinearEdgeProjection(nn.Module):
    """
    Projects RBF expanded edge features into the node embedding space using a linear layer.
    Crucially, no activation function is applied to preserve geometric signal fidelity.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class ReceiverAwareConv(MessagePassing):
    """
    Receiver-Aware Graph Convolution Layer.
    Concatenates Target(i), Source(j), and Edge(ij) features.
    Uses a Gated mechanism with Softplus for content and Sigmoid for gating.
    """

    def __init__(self, hidden_dim):
        super().__init__(aggr="add")  # Sum aggregation
        self.hidden_dim = hidden_dim

        # Input is concatenation of Target(h_i), Source(h_j), Edge(e_ij)
        # Each is hidden_dim. Total = 3 * hidden_dim
        input_dim = 3 * hidden_dim

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(input_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, hidden_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, hidden_dim] (Already projected)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features
        # x_j: source node features
        # edge_attr: projected edge features

        # Concatenate [Target || Source || Edge]
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Gating mechanism
        # Content: Softplus(Linear1(z))
        # Gate: Sigmoid(Linear2(z))
        content = F.softplus(self.lin1(z_ij))
        gate = torch.sigmoid(self.lin2(z_ij))

        return content * gate


class AdaptiveResidualBlock(nn.Module):
    """
    Stabilized Adaptive Residual Block.
    Update rule: h_{l+1} = Softplus(BatchNorm(Agg(m_ij) + (1 + epsilon) * h_l))
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.conv = ReceiverAwareConv(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        # Learnable epsilon initialized to 0
        self.epsilon = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Message passing (Aggregation)
        m = self.conv(x, edge_index, edge_attr)

        # Adaptive residual connection
        # (1 + epsilon) * x is the residual part
        res = (1.0 + self.epsilon) * x

        # Sum-Normalization: BN applied to the sum
        out = m + res
        out = self.bn(out)

        # Activation after normalization
        out = F.softplus(out)

        # Dropout
        out = self.dropout(out)

        return out


class LP_RA_CGN(nn.Module):
    """
    Linearly-Projected Receiver-Aware Crystal Graph Network.
    """

    def __init__(
        self,
        node_input_dim=100,  # Max atomic number roughly
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_rbf=config.NUM_RBF,
        cutoff=config.CUTOFF_RADIUS,
        dropout=config.DROPOUT,
    ):
        super().__init__()

        # Node embedding
        self.embedding = nn.Embedding(node_input_dim, hidden_dim)

        # Edge expansion and projection
        self.rbf = RBFExpansion(vmin=0, vmax=cutoff, bins=num_rbf)
        self.edge_proj = LinearEdgeProjection(num_rbf, hidden_dim)

        # Interaction backbone
        self.blocks = nn.ModuleList(
            [AdaptiveResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Decoupled heads
        # Formation energy head
        self.head_formation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Bandgap energy head
        self.head_bandgap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Embed nodes
        h = self.embedding(x)

        # Embed edges
        e_rbf = self.rbf(edge_attr)
        e = self.edge_proj(e_rbf)

        # Interaction blocks
        for block in self.blocks:
            h = block(h, edge_index, e)

        # Global pooling
        h_pool = global_mean_pool(h, batch)

        # Prediction heads
        out_formation = self.head_formation(h_pool)
        out_bandgap = self.head_bandgap(h_pool)

        # Concatenate outputs [batch_size, 2]
        return torch.cat([out_formation, out_bandgap], dim=1)


def train_model(
    model,
    train_loader,
    val_loader,
    target_scaler,
    device,
    epochs=config.MAX_EPOCHS,
    lr=config.LEARNING_RATE,
):
    """
    Training loop with validation and early stopping.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds = model(batch)

            # Transform targets using the scaler
            targets = batch.y
            if target_scaler is not None:
                targets = target_scaler.transform(targets)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch)
                targets = batch.y

                # Calculate loss in standardized space for scheduler/early stopping
                if target_scaler is not None:
                    targets_scaled = target_scaler.transform(targets)
                    loss = criterion(preds, targets_scaled)
                else:
                    loss = criterion(preds, targets)

                val_loss += loss.item() * batch.num_graphs

                # Inverse transform for RMSLE calculation
                if target_scaler is not None:
                    preds_original = target_scaler.inverse_transform(preds)
                else:
                    preds_original = preds

                val_preds_list.append(preds_original.cpu())
                val_targets_list.append(batch.y.cpu())

        val_loss /= len(val_loader.dataset)

        val_preds_all = torch.cat(val_preds_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)

        val_rmsle = utils.compute_rmsle(val_preds_all, val_targets_all)

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSLE: {val_rmsle:.10f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save checkpoint
            torch.save(
                best_model_state, os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def generate_submission(model, test_loader, target_scaler, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)

            if target_scaler is not None:
                preds = target_scaler.inverse_transform(preds)

            # Clamp negative predictions to 0 (physical constraint)
            preds = torch.clamp(preds, min=0.0)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(batch.id.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID
    df = df.sort_values("id")

    # Save
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_pipeline():
    """
    Main pipeline execution.
    """
    utils.set_seed(config.SEED)

    # Load data
    train_loader, val_loader, test_loader, target_scaler = data.get_dataloaders()

    # Initialize model
    # Atomic numbers up to ~100 (In is 49, Ga 31, Al 13, O 8). 100 is safe.
    model = LP_RA_CGN(node_input_dim=100).to(config.DEVICE)

    # Train
    model = train_model(model, train_loader, val_loader, target_scaler, config.DEVICE)

    # Predict
    generate_submission(model, test_loader, target_scaler, config.DEVICE)
