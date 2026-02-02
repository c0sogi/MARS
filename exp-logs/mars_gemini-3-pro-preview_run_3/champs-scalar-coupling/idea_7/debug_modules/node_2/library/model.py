import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch
import pandas as pd
import os
from library.config import Config


class CFConv(MessagePassing):
    """
    Continuous Filter Convolution.
    Generates filters from edge attributes using an MLP and aggregates neighbors.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="add")
        self.filter_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (N, node_dim)
        # edge_index: (2, E)
        # edge_attr: (E, edge_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Generate filter from edge attributes
        w = self.filter_net(edge_attr)
        # Element-wise modulation (SchNet-style)
        return x_j * w


class InteractionBlock(nn.Module):
    """
    Updates Line Graph (edges) and then Atom Graph (nodes).
    """

    def __init__(self, hidden_dim, rbf_angle_dim):
        super().__init__()

        # Line Graph Update:
        # Nodes are edges (dim=hidden_dim)
        # Edges are angles (dim=rbf_angle_dim)
        self.line_conv = CFConv(hidden_dim, rbf_angle_dim, hidden_dim)
        self.line_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Atom Graph Update:
        # Nodes are atoms (dim=hidden_dim)
        # Edges are the updated edge embeddings (dim=hidden_dim)
        self.atom_conv = CFConv(hidden_dim, hidden_dim, hidden_dim)
        self.atom_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, e, line_edge_index, line_edge_attr):
        """
        h: Atom features (N, hidden)
        e: Edge features (E, hidden)
        line_edge_index: Adjacency of line graph (2, E_line)
        line_edge_attr: Angle RBFs (E_line, rbf_angle)
        """
        # 1. Update Edge Embeddings (Line Graph Message Passing)
        if line_edge_index.numel() > 0:
            m_e = self.line_conv(e, line_edge_index, line_edge_attr)
            e = e + self.line_update(m_e)

        # 2. Update Atom Embeddings (Atom Graph Message Passing)
        # We use the updated 'e' as the edge attribute for the atom convolution
        m_h = self.atom_conv(h, edge_index, e)
        h = h + self.atom_update(m_h)

        return h, e


class SGLGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM

        # --- Embeddings ---
        # Atom types: H, C, N, O, F (5 types)
        self.atom_emb = nn.Embedding(5, self.hidden_dim)
        # Project distance RBFs to initial edge embeddings
        self.edge_emb = nn.Linear(Config.NUM_RBF_DIST, self.hidden_dim)

        # --- Interaction Blocks ---
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(self.hidden_dim, Config.NUM_RBF_ANGLE)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # --- Type-Specific Prediction Heads ---
        # Input: h_i (hidden) + h_j (hidden) + e_ij (hidden)
        self.heads = nn.ModuleDict()
        input_dim = self.hidden_dim * 3

        for t in Config.COUPLING_TYPES:
            self.heads[t] = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )

        # --- Auxiliary Heads ---
        # Magnetic Shielding (9 components)
        self.shield_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 9),
        )
        # Mulliken Charge (1 component)
        self.charge_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # --- Metadata Map ---
        # Used to reconstruct batch indices for targets
        self.mol_counts = self._build_mol_counts()

    def _build_mol_counts(self):
        """
        Pre-loads metadata to map molecule_name to number of targets.
        This is necessary because PyG Batch doesn't track which graph a target belongs to.
        """
        counts = {}
        # Check all possible metadata files
        paths = [Config.TRAIN_METADATA, Config.VAL_METADATA, Config.TEST_METADATA]

        print("S-GLGN: Building molecule target counts map...")
        for path in paths:
            if os.path.exists(path):
                # Only read necessary columns for speed
                try:
                    df = pd.read_csv(path, usecols=["molecule_name"])
                    c = df.groupby("molecule_name").size().to_dict()
                    counts.update(c)
                except Exception as e:
                    print(f"Warning: Could not read {path}: {e}")
        return counts

    def forward(self, data):
        # Unpack data
        x, batch = data.x, data.batch
        edge_index, edge_attr = data.edge_index, data.edge_attr
        line_edge_index, line_edge_attr = data.line_edge_index, data.line_edge_attr

        # 1. Initial Embeddings
        h = self.atom_emb(x)
        e = self.edge_emb(edge_attr)

        # 2. Message Passing
        for block in self.blocks:
            h, e = block(h, edge_index, e, line_edge_index, line_edge_attr)

        # 3. Auxiliary Predictions (Regularization)
        pred_shield = self.shield_head(h)
        pred_charge = self.charge_head(h).squeeze(-1)

        # 4. Readout for Coupling Constants
        # We need to gather h_i, h_j, and e_ij for the target pairs.

        # A. Construct 'target_batch' vector to map targets to graphs
        mol_names = data.molecule_name
        # Handle case where mol_name might be missing from map (safety)
        target_counts = [self.mol_counts.get(name, 0) for name in mol_names]

        # Create batch vector for targets: [0, 0, ..., 1, 1, ...]
        target_batch = torch.repeat_interleave(
            torch.arange(len(mol_names), device=x.device),
            torch.tensor(target_counts, device=x.device),
        )

        # B. Convert graph features to Dense Batch for efficient indexing
        # h_dense: (Batch, MaxNodes, Hidden)
        h_dense, mask = to_dense_batch(h, batch)

        # e_dense: (Batch, MaxNodes, MaxNodes, Hidden)
        # Puts 0 where edge doesn't exist (dist > cutoff)
        e_dense = to_dense_adj(
            edge_index, batch, edge_attr=e, max_num_nodes=h_dense.size(1)
        )

        # C. Gather Features
        # target_edge_index contains local atom indices (0..N-1)
        u_local = data.target_edge_index[0]
        v_local = data.target_edge_index[1]

        # Gather Atom Features
        h_u = h_dense[target_batch, u_local]
        h_v = h_dense[target_batch, v_local]

        # Gather Edge Features
        e_uv = e_dense[target_batch, u_local, v_local]

        # Concatenate: [h_u, h_v, e_uv]
        out_feat = torch.cat([h_u, h_v, e_uv], dim=-1)

        # D. Route to Type-Specific Heads
        preds = torch.zeros(len(target_batch), device=x.device)

        # Iterate over types present in this batch
        unique_types = torch.unique(data.target_type)
        for t_idx in unique_types:
            t_idx_item = t_idx.item()
            # Find name from index
            t_name = Config.COUPLING_TYPES[t_idx_item]

            # Mask for this type
            mask_t = data.target_type == t_idx

            # Pass through specific head
            # out_feat[mask_t] -> (Num_Type, InputDim)
            out_t = self.heads[t_name](out_feat[mask_t]).squeeze(-1)

            preds[mask_t] = out_t

        return preds, pred_shield, pred_charge
