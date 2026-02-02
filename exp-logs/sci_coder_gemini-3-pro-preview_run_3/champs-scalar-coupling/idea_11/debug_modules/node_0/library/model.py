import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Functions for expanding scalar distances/angles.
    """

    def __init__(self, start=0.0, end=5.0, num_basis=Config.NUM_RBF):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(start, end, num_basis))
        self.sigma = nn.Parameter(torch.ones(1) * (end - start) / num_basis)

    def forward(self, x):
        # x: (N,)
        # out: (N, num_basis)
        return torch.exp(-((x.unsqueeze(1) - self.centers) ** 2) / (2 * self.sigma**2))


class DMPNNLayer(nn.Module):
    """
    Directional Message Passing Layer.
    Updates edge embeddings based on interactions with preceding edges (triplets).
    """

    def __init__(self, hidden_dim, num_rbf_angle):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Interaction Block
        # Transforms source edge embedding and angle embedding
        self.lin_edge = nn.Linear(hidden_dim, hidden_dim)
        self.lin_angle = nn.Linear(num_rbf_angle, hidden_dim)

        # Update Block
        # Aggregates messages and updates target edge embedding
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_emb, triplets, angle_rbf):
        """
        Args:
            edge_emb: (E, hidden_dim) - Current edge embeddings
            triplets: (2, T) - Indices [incoming_edge_idx, outgoing_edge_idx]
            angle_rbf: (T, num_rbf_angle) - Expanded bond angles for triplets
        """
        idx_kj, idx_ji = triplets

        # 1. Compute Messages for Triplets (k -> j -> i)
        # Message = Transformation(e_kj) * Transformation(Angle_kji)
        # Hadamard product allows angle to gate/modulate the information flow
        msg_triplet = self.lin_edge(edge_emb[idx_kj]) * self.lin_angle(angle_rbf)

        # 2. Aggregate Messages to Target Edge (j -> i)
        # Sum all messages directed to the same outgoing edge
        aggr_msg = scatter_add(msg_triplet, idx_ji, dim=0, dim_size=edge_emb.size(0))

        # 3. Update Edge Embeddings (Residual Connection)
        out = edge_emb + self.update_mlp(aggr_msg)
        out = self.norm(out)

        return out


class ScalarCouplingModel(nn.Module):
    """
    Primary Model Architecture.

    Components:
    1. Node & Edge Initialization (Embedding + RBF)
    2. Directional Message Passing Backbone (DMPNN)
    3. Auxiliary Heads (Shielding, Charge) for Regularization
    4. Shared Conditional Readout Head with Edge Injection
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_rbf = Config.NUM_RBF
        self.num_abf = Config.NUM_ABF

        # 1. Initialization
        self.atom_emb = nn.Embedding(Config.NUM_ATOM_TYPES, self.hidden_dim)
        self.type_emb = nn.Embedding(Config.NUM_COUPLING_TYPES, self.hidden_dim)

        self.dist_rbf = GaussianRBF(
            start=0.0, end=Config.RADIUS_CUTOFF, num_basis=self.num_rbf
        )
        self.angle_rbf = GaussianRBF(
            start=-1.0, end=1.0, num_basis=self.num_abf
        )  # Cosine range [-1, 1]

        # Initial Edge Embedding: MLP(h_u || h_v || RBF(d_uv))
        self.edge_init = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.num_rbf, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 2. Backbone
        self.layers = nn.ModuleList(
            [DMPNNLayer(self.hidden_dim, self.num_abf) for _ in range(self.num_layers)]
        )

        # Node Update (Aggregating final edge embeddings back to nodes)
        self.node_update = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 3. Auxiliary Heads
        self.shielding_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 9),  # 3x3 Tensor flattened
        )
        self.charge_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),  # Scalar charge
        )

        # 4. Readout Head
        # Input: h_atom0 || h_atom1 || e_edge01 || type_embedding
        self.coupling_head = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def _calculate_angles(self, pos, edge_index, triplets):
        """
        Computes cosine of bond angles for all triplets.
        """
        idx_kj, idx_ji = triplets

        # Nodes involved in triplet k -> j -> i
        # edge_index is (source, target)
        k = edge_index[0, idx_kj]
        j = edge_index[1, idx_kj]  # == edge_index[0, idx_ji]
        i = edge_index[1, idx_ji]

        pos_k = pos[k]
        pos_j = pos[j]
        pos_i = pos[i]

        # Vectors pointing away from central atom j
        vec_jk = pos_k - pos_j
        vec_ji = pos_i - pos_j

        # Normalize
        vec_jk = F.normalize(vec_jk, dim=-1)
        vec_ji = F.normalize(vec_ji, dim=-1)

        # Cosine similarity
        cos_theta = (vec_jk * vec_ji).sum(dim=-1)
        return cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def _get_edge_embeddings_for_couplings(
        self, edge_emb, edge_index, coupling_atom_index, num_nodes
    ):
        """
        Retrieves the learned edge embedding for the specific atom pairs involved in coupling.
        If no direct edge exists (distance > cutoff), returns a zero vector.
        Uses hashing and binary search for efficiency on GPU.
        """
        # Create unique hash for edges: u * multiplier + v
        # Multiplier must be > max node index in batch
        multiplier = num_nodes + 1

        edge_hash = edge_index[0] * multiplier + edge_index[1]
        coup_hash = coupling_atom_index[0] * multiplier + coupling_atom_index[1]

        # Sort existing edges to enable binary search
        sorted_edge_hash, perm = torch.sort(edge_hash)

        # Find indices of coupling pairs in the sorted edge list
        idx = torch.searchsorted(sorted_edge_hash, coup_hash)

        # Handle out-of-bounds indices returned by searchsorted
        idx = idx.clamp(0, len(sorted_edge_hash) - 1)

        # Check which couplings actually matched an edge
        matched_hash = sorted_edge_hash[idx]
        mask = matched_hash == coup_hash

        # Initialize output with zeros
        selected_emb = torch.zeros(
            coupling_atom_index.size(1), self.hidden_dim, device=edge_emb.device
        )

        # Fill in embeddings for found edges
        # perm maps sorted index back to original edge_emb index
        valid_indices = perm[idx[mask]]
        selected_emb[mask] = edge_emb[valid_indices]

        return selected_emb

    def forward(self, data):
        """
        Forward pass.
        Args:
            data: PyG Data object containing batch of molecules.
        Returns:
            pred_coupling: (N_couplings,)
            pred_shielding: (N_nodes, 9)
            pred_charge: (N_nodes,)
        """
        x, pos, edge_index, edge_attr, triplets = (
            data.x,
            data.pos,
            data.edge_index,
            data.edge_attr,
            data.triplets,
        )
        coupling_atom_index, coupling_type = (
            data.coupling_atom_index,
            data.coupling_type,
        )

        # 1. Initial Node Embeddings
        h = self.atom_emb(x)

        # 2. Edge Initialization
        # edge_attr is [dist, vec_x, vec_y, vec_z]
        dist = edge_attr[:, 0]
        dist_emb = self.dist_rbf(dist)

        row, col = edge_index
        h_row = h[row]
        h_col = h[col]

        # Concatenate node features and distance to form initial edge state
        e = self.edge_init(torch.cat([h_row, h_col, dist_emb], dim=-1))

        # 3. Precompute Angle Features
        cos_theta = self._calculate_angles(pos, edge_index, triplets)
        angle_emb = self.angle_rbf(cos_theta)

        # 4. Directional Message Passing
        for layer in self.layers:
            e = layer(e, triplets, angle_emb)

        # 5. Node Update
        # Aggregate final edge embeddings to update node states for aux tasks
        # Edges are j->i (col is target), so we scatter_mean to col
        aggr_edges = scatter_mean(e, col, dim=0, dim_size=h.size(0))
        h_updated = h + self.node_update(torch.cat([h, aggr_edges], dim=-1))

        # 6. Auxiliary Predictions
        pred_shielding = self.shielding_head(h_updated)
        pred_charge = self.charge_head(h_updated)

        # 7. Coupling Prediction (Readout)
        # Gather node features for the coupling pairs
        idx_0, idx_1 = coupling_atom_index
        h_0 = h_updated[idx_0]
        h_1 = h_updated[idx_1]

        # Gather edge features (Edge Injection)
        # Injects the specific geometric environment of the bond if it exists
        e_01 = self._get_edge_embeddings_for_couplings(
            e, edge_index, coupling_atom_index, h.size(0)
        )

        # Get Coupling Type Embedding
        t_emb = self.type_emb(coupling_type)

        # Concatenate all information
        out_input = torch.cat([h_0, h_1, e_01, t_emb], dim=-1)

        # Predict
        pred_coupling = self.coupling_head(out_input)

        return pred_coupling.squeeze(-1), pred_shielding, pred_charge.squeeze(-1)
