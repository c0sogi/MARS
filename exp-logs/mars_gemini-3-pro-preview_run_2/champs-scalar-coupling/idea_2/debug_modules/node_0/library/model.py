import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from library.config import Config
from library.layers import RadialBasis, AngularBasis, MLP


class EmbeddingBlock(nn.Module):
    """
    Initializes directional edge embeddings based on source/target atom types
    and the scalar distance between them.
    """

    def __init__(self, hidden_dim=Config.HIDDEN_DIM):
        super(EmbeddingBlock, self).__init__()
        self.atom_embedding = nn.Embedding(Config.NUM_ATOM_TYPES, hidden_dim)
        self.rbf = RadialBasis(Config.NUM_RBF, Config.CUTOFF_RADIUS)

        # Project RBF features to hidden dimension
        self.edge_embedding = nn.Linear(Config.NUM_RBF, hidden_dim)

        # MLP to combine [h_source, h_target, e_distance] into initial edge state m_ji
        self.init_mlp = MLP(
            in_dim=3 * hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
        )

    def forward(self, atom_types, edge_index, edge_dists):
        """
        Args:
            atom_types: (N,) Atom type indices
            edge_index: (2, E) Edge indices [src, dst]
            edge_dists: (E,) Distances
        Returns:
            m: (E, H) Initial directional edge embeddings
            edge_feat: (E, H) Static edge distance features
        """
        # 1. Embed Atoms
        h = self.atom_embedding(atom_types)  # (N, H)

        # 2. Embed Distances
        rbf_feat = self.rbf(edge_dists)  # (E, RBF)
        edge_feat = self.edge_embedding(rbf_feat)  # (E, H)

        # 3. Create Directional Embeddings
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # (E, H)
        h_dst = h[dst]  # (E, H)

        # Concatenate source atom, target atom, and edge geometry
        concat_feat = torch.cat([h_src, h_dst, edge_feat], dim=-1)

        # Project to hidden state
        m = self.init_mlp(concat_feat)  # (E, H)

        return m, edge_feat


class InteractionBlock(nn.Module):
    """
    Performs directional message passing.
    Updates edge embedding m_{j->i} by aggregating messages from m_{k->j}.
    Messages are modulated by the angle k-j-i.
    """

    def __init__(self, hidden_dim=Config.HIDDEN_DIM):
        super(InteractionBlock, self).__init__()
        self.abf = AngularBasis(Config.NUM_ABF)

        # MLP to compute messages from triplets: [m_kj, e_kj, angular_basis] -> message
        self.message_mlp = MLP(
            in_dim=2 * hidden_dim + Config.NUM_ABF,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
        )

        # MLP to update edge states: [m_ji_old, aggregated_messages, e_ji] -> m_ji_new
        self.update_mlp = MLP(
            in_dim=3 * hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, m, edge_feat, triplet_edge_index, triplet_angles, num_edges):
        """
        Args:
            m: (E, H) Current edge embeddings
            edge_feat: (E, H) Static edge distance features
            triplet_edge_index: (2, T) [incoming_edge_idx, outgoing_edge_idx]
            triplet_angles: (T,) Angles in radians
            num_edges: Total number of edges (for scatter dimension)
        Returns:
            m_new: (E, H) Updated edge embeddings
        """
        # 1. Compute Angular Features
        abf_feat = self.abf(triplet_angles)  # (T, ABF)

        # 2. Gather Incoming Edge Features
        # triplet_edge_index[0] corresponds to edge k->j
        in_edge_idx = triplet_edge_index[0]
        m_in = m[in_edge_idx]  # (T, H)
        e_in = edge_feat[in_edge_idx]  # (T, H)

        # 3. Compute Triplet Messages
        # Combine incoming edge state, incoming edge geometry, and angle geometry
        triplet_input = torch.cat([m_in, e_in, abf_feat], dim=-1)
        triplet_msg = self.message_mlp(triplet_input)  # (T, H)

        # 4. Aggregate Messages to Outgoing Edges
        # triplet_edge_index[1] corresponds to edge j->i
        out_edge_idx = triplet_edge_index[1]

        # Sum all messages directed towards specific edges
        agg_msg = scatter_sum(
            triplet_msg, out_edge_idx, dim=0, dim_size=num_edges
        )  # (E, H)

        # 5. Update Edge States
        # Combine old state, aggregated messages, and local edge geometry
        update_input = torch.cat([m, agg_msg, edge_feat], dim=-1)
        m_update = self.update_mlp(update_input)

        # Residual connection + LayerNorm
        m_new = self.layer_norm(m + m_update)

        return m_new


class DMPNN(nn.Module):
    """
    Directional Message Passing Neural Network.
    """

    def __init__(self):
        super(DMPNN, self).__init__()

        self.embedding = EmbeddingBlock(Config.HIDDEN_DIM)

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(Config.HIDDEN_DIM)
                for _ in range(Config.NUM_INTERACTIONS)
            ]
        )

        self.type_embedding = nn.Embedding(Config.NUM_COUPLING_TYPES, Config.HIDDEN_DIM)

        # Readout Head: [m_fwd, m_bwd, type_emb] -> Scalar Coupling Constant
        self.readout_mlp = MLP(
            in_dim=3 * Config.HIDDEN_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            out_dim=1,
            num_layers=3,
        )

    def forward(self, data):
        """
        Args:
            data: Batch dictionary containing graph data.
        Returns:
            out: (B, 1) Predicted scalar coupling constants.
        """
        atom_types = data["atom_types"]
        edge_index = data["edge_index"]
        edge_dists = data["edge_dists"]
        triplet_edge_index = data["triplet_edge_index"]
        triplet_angles = data["triplet_angles"]
        type_idxs = data["type_idxs"]
        target_edge_indices = data["target_edge_indices"]  # (B, 2)

        num_edges = edge_index.shape[1]

        # 1. Initial Embedding
        # m: (E, H), edge_feat: (E, H)
        m, edge_feat = self.embedding(atom_types, edge_index, edge_dists)

        # 2. Interaction Layers
        for layer in self.interactions:
            m = layer(m, edge_feat, triplet_edge_index, triplet_angles, num_edges)

        # 3. Readout
        # Extract embeddings for the target atom pairs
        # target_edge_indices contains [idx_0->1, idx_1->0]
        idx_fwd = target_edge_indices[:, 0]
        idx_bwd = target_edge_indices[:, 1]

        # Handle cases where edges might be missing (-1) by clamping to 0
        # (The dataset generator uses a generous cutoff, so missing edges should be rare/impossible for bonded pairs)
        idx_fwd = idx_fwd.clamp(min=0)
        idx_bwd = idx_bwd.clamp(min=0)

        m_fwd = m[idx_fwd]  # (B, H)
        m_bwd = m[idx_bwd]  # (B, H)

        # Get coupling type embedding
        t_emb = self.type_embedding(type_idxs)  # (B, H)

        # Concatenate pair representations and type info
        out_input = torch.cat([m_fwd, m_bwd, t_emb], dim=-1)

        # Predict
        out = self.readout_mlp(out_input)  # (B, 1)

        return out
