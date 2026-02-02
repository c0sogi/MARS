import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from library.config import Config
from library.features import GaussianRBF, AngleRBF


class InteractionBlock(nn.Module):
    """
    Directional Message Passing Interaction Block.

    Updates directed edge embeddings by aggregating messages from incoming edges (k->j)
    to the outgoing edge (j->i). Explicitly models the bond angle theta_kji.
    """

    def __init__(self, hidden_dim, num_angle_rbf, rbf_gamma, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Angular features
        self.angle_rbf = AngleRBF(
            start=-1.0, stop=1.0, num_rbf=num_angle_rbf, gamma=rbf_gamma
        )

        # Transformation layers
        self.linear_msg = nn.Linear(hidden_dim, hidden_dim)
        self.linear_angle = nn.Linear(num_angle_rbf, hidden_dim)

        # Update network (ResNet-style)
        self.linear_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_edge, triplet_index, edge_vec):
        """
        Args:
            h_edge: (E, hidden_dim) Edge embeddings.
            triplet_index: (3, T) Indices [edge_in, edge_out, center_node].
            edge_vec: (E, 3) Vector pointing from src to dst for each edge.
        """
        # Unpack indices
        idx_in, idx_out = triplet_index[0], triplet_index[1]

        # 1. Compute Geometric Features (Angles)
        # Vector k->j (incoming)
        v_in = edge_vec[idx_in]
        # Vector j->i (outgoing)
        v_out = edge_vec[idx_out]

        # Normalize vectors
        v_in = F.normalize(v_in, dim=-1)
        v_out = F.normalize(v_out, dim=-1)

        # Calculate Cosine Angle
        # Note: edge_vec is defined as (pos_dst - pos_src).
        # For triplet k->j->i:
        #   Incoming edge k->j has vector (pos_j - pos_k).
        #   Outgoing edge j->i has vector (pos_i - pos_j).
        #   The bond angle is between vector j->k (which is -v_in) and j->i (v_out).
        #   Dot product: (-v_in) . v_out
        cos_theta = -(v_in * v_out).sum(dim=-1, keepdim=True)  # (T, 1)

        # Expand angle using RBF
        rbf_angle = self.angle_rbf(cos_theta)  # (T, num_angle_rbf)

        # 2. Compute Messages
        # Transform incoming edge features
        msg_node = self.linear_msg(h_edge[idx_in])
        # Transform angular features
        msg_angle = self.linear_angle(rbf_angle)
        # Modulate message by angle (Hadamard product)
        msg = msg_node * msg_angle

        # 3. Aggregate Messages
        # Sum all messages directed to the same outgoing edge
        # dim_size ensures we cover all edges even if some receive no messages
        agg_msg = scatter_add(msg, idx_out, dim=0, dim_size=h_edge.size(0))

        # 4. Update Edge State
        # Concatenate original state with aggregated messages
        cat_feat = torch.cat([h_edge, agg_msg], dim=-1)
        update = self.linear_update(cat_feat)

        # Residual connection + Norm
        return self.norm(h_edge + update)


class DMPNN(nn.Module):
    """
    Directional Message Passing Neural Network with Shared Conditional Readout.

    Features:
    - Continuous Filter Convolutions for distances and angles.
    - Edge-centric message passing.
    - Auxiliary heads for Charge and Shielding prediction.
    - Shared readout head conditioned on coupling type.
    """

    def __init__(
        self,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_rbf=Config.NUM_RBF,
        num_angle_rbf=Config.NUM_ANGLE_RBF,
        rbf_gamma=Config.RBF_GAMMA,
        dropout=Config.DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ==========================
        # 1. Embedding Layers
        # ==========================
        self.node_emb = nn.Embedding(Config.NUM_ATOM_TYPES, hidden_dim)
        self.coupling_emb = nn.Embedding(Config.NUM_COUPLING_TYPES, hidden_dim)

        # Distance Expansion
        self.dist_rbf = GaussianRBF(
            start=0.0, stop=Config.MAX_RADIUS, num_rbf=num_rbf, gamma=rbf_gamma
        )

        # Initial Edge Embedding Network
        # Inputs: [RBF(dist), NodeEmb(src), NodeEmb(dst)]
        self.edge_init = nn.Sequential(
            nn.Linear(num_rbf + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ==========================
        # 2. Interaction Backbone
        # ==========================
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(hidden_dim, num_angle_rbf, rbf_gamma, dropout)
                for _ in range(num_layers)
            ]
        )

        # ==========================
        # 3. Node Aggregation
        # ==========================
        # Aggregates edge features back to nodes after message passing
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ==========================
        # 4. Prediction Heads
        # ==========================

        # Main Task: Scalar Coupling Prediction
        # Input: [h_i, h_j, RBF(dist_ij), CouplingTypeEmb]
        coupling_input_dim = (hidden_dim * 2) + num_rbf + hidden_dim
        self.head_coupling = nn.Sequential(
            nn.Linear(coupling_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Aux Task: Mulliken Charge (Scalar)
        self.head_charge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Aux Task: Magnetic Shielding (Tensor components -> 9 values)
        self.head_shielding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 9),
        )

    def forward(self, batch):
        """
        Args:
            batch (dict): Batch dictionary from SoACollator.
        Returns:
            dict: Predictions for 'coupling', 'charge', 'shielding'.
        """
        # Unpack batch
        pos = batch["pos"]
        node_type = batch["node_type"]
        edge_index = batch["edge_index"]
        edge_vec = batch["edge_vec"]
        edge_dist = batch["edge_dist"]
        triplet_index = batch["triplet_index"]

        # ------------------------------------------------
        # 1. Initialization
        # ------------------------------------------------
        # Node Embeddings
        h_node = self.node_emb(node_type)  # (N, dim)

        # Edge Embeddings
        # Construct initial features from geometry and source/dest nodes
        src, dst = edge_index[0], edge_index[1]
        rbf_dist = self.dist_rbf(edge_dist)
        h_src = h_node[src]
        h_dst = h_node[dst]

        # (E, num_rbf + 2*dim) -> (E, dim)
        h_edge = self.edge_init(torch.cat([rbf_dist, h_src, h_dst], dim=-1))

        # ------------------------------------------------
        # 2. Message Passing (Directional)
        # ------------------------------------------------
        for block in self.blocks:
            h_edge = block(h_edge, triplet_index, edge_vec)

        # ------------------------------------------------
        # 3. Node Aggregation
        # ------------------------------------------------
        # Sum all incoming edge features to update node representations
        # h_i_new = Update(h_i, sum_{j->i} h_{j->i})
        agg_edges = scatter_add(h_edge, dst, dim=0, dim_size=h_node.size(0))
        h_node_final = self.node_update(torch.cat([h_node, agg_edges], dim=-1))

        # ------------------------------------------------
        # 4. Predictions
        # ------------------------------------------------

        # --- Auxiliary Tasks ---
        pred_charge = self.head_charge(h_node_final).squeeze(-1)
        pred_shielding = self.head_shielding(h_node_final)

        # --- Main Task: Scalar Coupling ---
        # Retrieve indices for coupling pairs
        c_idxs = batch["coupling_atom_index"]  # (2, C)
        c_type = batch["coupling_type"]  # (C,)

        idx0, idx1 = c_idxs[0], c_idxs[1]

        # Gather Node Features
        h0 = h_node_final[idx0]
        h1 = h_node_final[idx1]

        # Get Coupling Type Embedding
        type_emb = self.coupling_emb(c_type)

        # Re-compute distance for the specific coupling pairs
        # (Note: These pairs might not correspond exactly to edges in the radius graph,
        # so we compute distance from positions directly)
        diff_vec = pos[idx0] - pos[idx1]
        dist_coupling = diff_vec.norm(dim=-1)
        dist_feat = self.dist_rbf(dist_coupling)

        # Concatenate all features
        # [h_i, h_j, dist_ij, type_emb]
        out_feat = torch.cat([h0, h1, dist_feat, type_emb], dim=-1)

        # Predict
        pred_coupling = self.head_coupling(out_feat).squeeze(-1)

        return {
            "coupling": pred_coupling,
            "charge": pred_charge,
            "shielding": pred_shielding,
        }
