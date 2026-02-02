import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_batch

from library.config import Config
from library.data import get_dataloaders, COUPLING_TYPES
from library.utils import compute_log_mae, set_seed

# ==========================================
# Helper Modules
# ==========================================


class DMPNNLayer(nn.Module):
    """
    Directional Message Passing Layer with Geometric Interaction.
    Updates edge embeddings based on neighboring edges and triplet geometry.
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Interaction: Combine incoming edge (h_ji) and geometry (sbf_jik)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update: Combine self (h_ik) and aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_emb, triplet_emb, triplet_edge_indices, num_edges):
        """
        Args:
            edge_emb: [E, H]
            triplet_emb: [T, H] (Projected SBF features)
            triplet_edge_indices: Tuple (idx_ji, idx_ik) mapping triplets to edges.
            num_edges: Total number of edges E.
        """
        idx_ji, idx_ik = triplet_edge_indices

        # 1. Compute Messages for each triplet
        # Message from edge j->i to edge i->k mediated by angle jik
        h_ji = edge_emb[idx_ji]

        # Concatenate incoming edge state with geometric feature
        # Input: [T, 2*H] -> Output: [T, H]
        raw_msg = torch.cat([h_ji, triplet_emb], dim=-1)
        msg = self.msg_mlp(raw_msg)

        # 2. Aggregate messages to the target edge (i->k)
        # Sum all messages directed to edge ik
        agg_msg = scatter_sum(msg, idx_ik, dim=0, dim_size=num_edges)

        # 3. Update edge states
        # Residual connection + LayerNorm
        # Input: [E, 2*H] -> Output: [E, H]
        update_in = torch.cat([edge_emb, agg_msg], dim=-1)
        update_out = self.update_mlp(update_in)

        return self.norm(edge_emb + update_out)


class TransformerGlobalContext(nn.Module):
    """
    Global Context Module using Transformer Encoder.
    Aggregates edge features to nodes, applies Self-Attention, returns node features.
    """

    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.0):
        super().__init__()

        self.node_agg_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, edge_emb, edge_index, batch_idx, num_nodes):
        """
        Args:
            edge_emb: [E, H]
            edge_index: [2, E]
            batch_idx: [N] (Node to batch assignment)
            num_nodes: Total nodes N
        """
        # 1. Aggregate edges to nodes (incoming edges)
        # edge_index[1] is the target node of the edge
        node_emb = scatter_sum(edge_emb, edge_index[1], dim=0, dim_size=num_nodes)
        node_emb = self.node_agg_norm(node_emb)

        # 2. Prepare for Transformer (Dense Batching)
        # Convert [N, H] to [B, Max_Nodes, H] with mask
        x_dense, mask = to_dense_batch(node_emb, batch_idx)

        # 3. Apply Transformer
        # mask is [B, Max_Nodes] (True for real nodes, False for padding)
        # Transformer expects src_key_padding_mask where True is padded (ignored)
        # to_dense_batch returns mask where True is real. So we invert.
        padding_mask = ~mask

        x_trans = self.transformer(x_dense, src_key_padding_mask=padding_mask)

        # 4. Flatten back to [N, H]
        # Select only valid nodes
        x_flat = x_trans[mask]

        return x_flat


# ==========================================
# Main Model
# ==========================================


class HGANet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM

        # --- Embeddings ---
        # Atom types (H, C, N, O, F) -> 5 types
        self.atom_emb = nn.Embedding(5, self.hidden_dim)

        # Coupling types (8 types)
        self.type_emb = nn.Embedding(8, self.hidden_dim)

        # Geometric Feature Projections
        self.rbf_proj = nn.Sequential(
            nn.Linear(config.RBF_SIZE, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.sbf_proj = nn.Sequential(
            nn.Linear(config.SBF_SIZE, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Initial Edge State: Combine Atom Embs + RBF
        self.edge_init = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # src_atom, dst_atom, rbf
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # --- Backbone (DMPNN) ---
        self.layers = nn.ModuleList(
            [
                DMPNNLayer(self.hidden_dim, config.DROPOUT)
                for _ in range(config.NUM_MPNN_LAYERS)
            ]
        )

        # --- Global Context ---
        self.global_context = TransformerGlobalContext(
            self.hidden_dim,
            config.NUM_TRANSFORMER_LAYERS,
            config.NUM_HEADS,
            config.DROPOUT,
        )

        # --- Readout ---
        # Concatenation: Node_u + Node_v + Edge_uv + Type_Emb
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def _get_edge_lookup_indices(self, edge_index, query_pairs, num_nodes):
        """
        Maps query node pairs (u, v) to indices in the edge_index tensor.
        Uses hashing for O(N log N) efficiency.
        """
        # Create unique hash for edges: u * multiplier + v
        # Multiplier must be > max node index in the batch
        multiplier = num_nodes + 100

        # Hash existing edges
        # edge_index: [2, E]
        edge_hashes = edge_index[0] * multiplier + edge_index[1]

        # Sort hashes to allow binary search
        sorted_hashes, perm = torch.sort(edge_hashes)

        # Hash query pairs
        # query_pairs: [2, Q]
        query_hashes = query_pairs[0] * multiplier + query_pairs[1]

        # Find insertion points
        indices = torch.searchsorted(sorted_hashes, query_hashes)

        # Clamp to valid range
        indices = indices.clamp(max=len(sorted_hashes) - 1)

        # Check if hash actually matches (edge exists)
        found_hashes = sorted_hashes[indices]
        mask = found_hashes == query_hashes

        # Map back to original edge indices
        real_indices = perm[indices]

        return real_indices, mask

    def forward(self, data):
        x = data["x"]
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        triplet_index = data["triplet_index"]
        triplet_attr = data["triplet_attr"]
        batch = data["batch"]
        coup_index = data["coupling_index"]  # [C, 2]
        coup_type = data["coupling_type"]  # [C]

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # 1. Initialize Features
        h_nodes = self.atom_emb(x)  # [N, H]
        h_rbf = self.rbf_proj(edge_attr)  # [E, H]

        # Edge Init: [h_u, h_v, rbf]
        src, dst = edge_index
        h_edges = torch.cat([h_nodes[src], h_nodes[dst], h_rbf], dim=-1)
        h_edges = self.edge_init(h_edges)  # [E, H]

        # 2. Prepare Triplet Indices for DMPNN
        # Map triplet (j, i, k) to edge indices (j->i) and (i->k)
        if triplet_index.size(1) > 0:
            h_sbf = self.sbf_proj(triplet_attr)  # [T, H]

            # Query pairs: (j, i) and (i, k)
            pairs_ji = triplet_index[[0, 1], :]
            pairs_ik = triplet_index[[1, 2], :]

            # We assume edges exist for triplets by construction, but use safe lookup
            idx_ji, mask_ji = self._get_edge_lookup_indices(
                edge_index, pairs_ji, num_nodes
            )
            idx_ik, mask_ik = self._get_edge_lookup_indices(
                edge_index, pairs_ik, num_nodes
            )

            # Filter valid triplets (should be all, but for safety)
            valid_triplets = mask_ji & mask_ik
            triplet_edge_indices = (idx_ji[valid_triplets], idx_ik[valid_triplets])
            h_sbf = h_sbf[valid_triplets]
        else:
            # Handle molecules with no triplets (e.g. diatomics)
            h_sbf = torch.empty(0, self.hidden_dim, device=x.device)
            triplet_edge_indices = (
                torch.empty(0, dtype=torch.long, device=x.device),
                torch.empty(0, dtype=torch.long, device=x.device),
            )

        # 3. DMPNN Layers
        for layer in self.layers:
            h_edges = layer(h_edges, h_sbf, triplet_edge_indices, num_edges)

        # 4. Global Context (Transformer)
        h_global = self.global_context(h_edges, edge_index, batch, num_nodes)  # [N, H]

        # 5. Readout
        # Get features for coupling pairs (u, v)
        u_idx, v_idx = coup_index[:, 0], coup_index[:, 1]

        h_u = h_global[u_idx]
        h_v = h_global[v_idx]
        h_type = self.type_emb(coup_type)

        # Retrieve local edge feature e_{u->v}
        # Note: Edge might not exist if distance > cutoff
        pairs_uv = torch.stack([u_idx, v_idx], dim=0)
        idx_uv, mask_uv = self._get_edge_lookup_indices(edge_index, pairs_uv, num_nodes)

        # Create placeholder for missing edges
        e_uv = (
            torch.zeros_like(h_edges[0]).unsqueeze(0).expand(len(coup_type), -1).clone()
        )
        # Fill existing edges
        if mask_uv.any():
            e_uv[mask_uv] = h_edges[idx_uv[mask_uv]]

        # Concatenate and Predict
        out_vec = torch.cat([h_u, h_v, e_uv, h_type], dim=-1)
        pred = self.readout_mlp(out_vec)

        return pred.squeeze(-1)


# ==========================================
# Training & Submission Functions
# ==========================================


def train_model(config: Config):
    """
    Trains the HGA-Net model.
    """
    set_seed(config.SEED)

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, standardizer = get_dataloaders(
        config, load_cached_data=True
    )

    # 2. Model
    print("Initializing Model...")
    model = HGANet(config).to(config.DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    total_steps = len(train_loader) * config.EPOCHS

    # Ensure pct_start is valid (0 < pct_start < 1)
    pct_start = float(config.WARMUP_EPOCHS) / config.EPOCHS
    if pct_start >= 1.0 or pct_start <= 0.0:
        pct_start = 0.3

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    criterion = nn.L1Loss()

    best_metric = float("inf")
    patience = 5
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs on {config.DEVICE}...")

    for epoch in range(config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {
                k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch["y"])

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch["y"].size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []
        val_types = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                preds = model(batch)

                # Collect for metric calculation
                val_preds.append(preds.cpu())
                val_targets.append(batch["y"].cpu())
                val_types.append(batch["coupling_type"].cpu())

        # Concatenate
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_types = torch.cat(val_types).numpy()

        # Inverse Transform for Metric
        orig_preds = standardizer.inverse_transform(val_preds, val_types)
        orig_targets = standardizer.inverse_transform(val_targets, val_types)

        # Compute Metric
        # Map integer types back to strings for compute_log_mae
        type_str_map = np.array(COUPLING_TYPES)
        val_types_str = type_str_map[val_types]

        metric = compute_log_mae(orig_preds, orig_targets, val_types_str)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train MAE: {avg_train_loss:.6f} | Val LMAE: {metric}"
        )

        # Checkpointing
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  New best model saved! ({metric})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break


def generate_submission(config: Config):
    """
    Generates submission file using the best trained model.
    """
    set_seed(config.SEED)

    # 1. Load Data
    print("Loading test data...")
    _, _, test_loader, standardizer = get_dataloaders(config, load_cached_data=True)

    # 2. Load Model
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    model = HGANet(config).to(config.DEVICE)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.eval()

    # 3. Predict
    all_preds = []
    all_types = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = model(batch)
            all_preds.append(preds.cpu())
            all_types.append(batch["coupling_type"].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_types = torch.cat(all_types).numpy()

    # 4. Inverse Transform
    final_preds = standardizer.inverse_transform(all_preds, all_types)

    # 5. Align with ID
    # The dataloader yields predictions in the order of molecules in test_metadata.csv
    # We load metadata to get the IDs
    df_test = pd.read_csv(config.TEST_METADATA)

    if len(df_test) != len(final_preds):
        raise ValueError(
            f"Mismatch: {len(df_test)} metadata rows vs {len(final_preds)} predictions."
        )

    df_test["scalar_coupling_constant"] = final_preds

    # 6. Save
    submission_df = df_test[["id", "scalar_coupling_constant"]]
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
