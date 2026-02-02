import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands scalar values (distances or angles) into a vector of Radial Basis Functions.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Width of the Gaussians
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [N] -> [N, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    A single layer of the Dual-Graph Network.
    Updates Line Graph (bonds) based on angles, then Atom Graph (nodes) based on bonds.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Line Graph Update (Edge-to-Edge via Angles) ---
        # Input: [edge_i || edge_j || angle_feat]
        self.line_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update function for edges
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # --- Atom Graph Update (Node-to-Node via Edges) ---
        # Input: [node_i || node_j || edge_feat]
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update function for nodes
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, line_edge_index, line_edge_attr):
        """
        x: Node features [Num_Atoms, Hidden]
        edge_index: Atom graph edges [2, Num_Edges]
        edge_attr: Bond features [Num_Edges, Hidden]
        line_edge_index: Line graph edges (adjacency of bonds) [2, Num_Angle_Triplets]
        line_edge_attr: Angle features [Num_Angle_Triplets, Hidden]
        """

        # 1. Update Edge Features (Line Graph Interaction)
        # We propagate messages between bonds that share an atom (defined by line_edge_index)
        if line_edge_index.numel() > 0:
            src_bond, dst_bond = line_edge_index

            # Gather features of interacting bonds
            e_src = edge_attr[src_bond]
            e_dst = edge_attr[dst_bond]

            # Compute message based on the two bonds and the angle between them
            line_msg = self.line_mlp(torch.cat([e_src, e_dst, line_edge_attr], dim=-1))

            # Aggregate messages to the target bond
            line_aggr = scatter(
                line_msg, dst_bond, dim=0, dim_size=edge_attr.size(0), reduce="add"
            )

            # Residual update of edge features
            edge_attr_new = self.edge_update_mlp(
                torch.cat([edge_attr, line_aggr], dim=-1)
            )
            edge_attr = self.edge_norm(edge_attr + edge_attr_new)

        # 2. Update Node Features (Atom Graph Interaction)
        src_node, dst_node = edge_index

        x_src = x[src_node]
        x_dst = x[dst_node]

        # Compute message based on source node, destination node, and the connecting bond
        atom_msg = self.atom_mlp(torch.cat([x_src, x_dst, edge_attr], dim=-1))

        # Aggregate messages to the target node
        atom_aggr = scatter(atom_msg, dst_node, dim=0, dim_size=x.size(0), reduce="add")

        # Residual update of node features
        x_new = self.node_update_mlp(torch.cat([x, atom_aggr], dim=-1))
        x = self.node_norm(x + x_new)

        return x, edge_attr


class DualGraphNetwork(nn.Module):
    """
    Physics-Calibrated Dual-Graph Network with Edge-Centric Readout.
    """

    def __init__(self):
        super().__init__()

        hidden_dim = Config.HIDDEN_DIM
        num_layers = Config.NUM_LAYERS
        dropout = Config.DROPOUT

        # --- Embeddings & Feature Expansion ---
        self.node_emb = nn.Embedding(len(Config.ATOM_TYPES), hidden_dim)

        # Distance RBF (0 to Cutoff)
        self.dist_rbf = GaussianSmearing(
            start=0.0, stop=Config.RBF_CUTOFF, num_gaussians=Config.NUM_RBF
        )
        self.dist_proj = nn.Linear(Config.NUM_RBF, hidden_dim)

        # Angle RBF (-1 to 1 for Cosine)
        self.angle_rbf = GaussianSmearing(
            start=-1.0, stop=1.0, num_gaussians=Config.NUM_ANGLE_RBF
        )
        self.angle_proj = nn.Linear(Config.NUM_ANGLE_RBF, hidden_dim)

        # --- Backbone ---
        self.layers = nn.ModuleList(
            [InteractionBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # --- Auxiliary Heads (Physics Regularization) ---
        # Predicts Magnetic Shielding Tensor (flattened 3x3=9 values)
        self.aux_shielding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 9)
        )
        # Predicts Mulliken Charge (scalar)
        self.aux_charge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

        # --- Type-Specific Prediction Heads ---
        # Input: Node_i (Hidden) + Node_j (Hidden) + Edge_ij (Hidden)
        self.coupling_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(len(Config.COUPLING_TYPES))
            ]
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        line_edge_index = data.line_edge_index
        line_edge_attr = data.line_edge_attr

        # 1. Initialize Embeddings
        h = self.node_emb(x)

        # 2. Expand Geometric Features
        # Distances
        e_feat = self.dist_rbf(edge_attr.squeeze(-1))
        e_feat = self.dist_proj(e_feat)

        # Angles
        if line_edge_attr.numel() > 0:
            a_feat = self.angle_rbf(line_edge_attr.squeeze(-1))
            a_feat = self.angle_proj(a_feat)
        else:
            a_feat = torch.empty((0, Config.HIDDEN_DIM), device=x.device)

        # 3. Message Passing (Dual Graph Updates)
        for layer in self.layers:
            h, e_feat = layer(h, edge_index, e_feat, line_edge_index, a_feat)

        # 4. Auxiliary Predictions (Per Atom)
        pred_shielding = self.aux_shielding(h)
        pred_charge = self.aux_charge(h)

        # 5. Coupling Prediction (Per Pair)
        c_idx = data.coupling_atom_index  # [2, Num_Couplings]
        c_types = data.coupling_type  # [Num_Couplings]

        # Gather Node Features for the pair
        h_i = h[c_idx[0]]
        h_j = h[c_idx[1]]

        # Gather Edge Features for the pair (Edge-Centric Readout)
        # We need to find the specific edge index k in 'edge_index' that corresponds
        # to the coupling pair (u, v) to retrieve the updated e_feat[k].

        # Create unique hashes for edges to perform fast lookup
        # Hash = u * max_nodes + v
        max_nodes = h.size(0)
        edge_hash = edge_index[0].long() * max_nodes + edge_index[1].long()
        target_hash = c_idx[0].long() * max_nodes + c_idx[1].long()

        # Sort edge hashes to allow binary search
        sorted_hash, perm = torch.sort(edge_hash)

        # Find indices of targets in the sorted edge list
        idx_in_sorted = torch.searchsorted(sorted_hash, target_hash)

        # Clamp to valid range (handle case where target > all edges)
        idx_in_sorted = torch.clamp(idx_in_sorted, max=sorted_hash.size(0) - 1)

        # Verify matches (filter out non-existent edges, e.g., > cutoff)
        found_mask = sorted_hash[idx_in_sorted] == target_hash

        # Map back to original edge indices
        edge_indices = perm[idx_in_sorted]

        # Retrieve features
        # Initialize with zeros for pairs not found in the radius graph
        batch_e_feat = torch.zeros(
            (c_idx.size(1), e_feat.size(1)), device=e_feat.device, dtype=e_feat.dtype
        )

        if found_mask.any():
            batch_e_feat[found_mask] = e_feat[edge_indices[found_mask]]

        # Concatenate: [h_i || h_j || e_{ij}]
        out_feat = torch.cat([h_i, h_j, batch_e_feat], dim=-1)

        # Apply Type-Specific Heads
        pred_coupling = torch.zeros_like(data.coupling_value)

        for t_idx in range(len(Config.COUPLING_TYPES)):
            # Mask for current coupling type
            mask = c_types == t_idx
            if mask.any():
                # Predict
                pred_coupling[mask] = self.coupling_heads[t_idx](
                    out_feat[mask]
                ).squeeze(-1)

        return pred_coupling, pred_shielding, pred_charge
