import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_batch

from library.config import Config
from library.layers import RBFExpansion, SBFExpansion, MLP, TransformerBlock, DMPNNLayer


class HGANet(nn.Module):
    """
    Hybrid Geometric-Attention Network (HGA-Net)

    Architecture:
    1. Local Geometric Encoder: Directional Message Passing (DMPNN) with RBF/SBF basis functions.
    2. Global Interaction Module: Transformer Encoder on atom-level embeddings.
    3. Deterministic Readout: Pairwise regression MLP.
    """

    def __init__(self):
        super(HGANet, self).__init__()

        # ---------------------------------------------------------------------
        # Hyperparameters
        # ---------------------------------------------------------------------
        hidden_dim = Config.HIDDEN_DIM
        rbf_size = Config.RBF_SIZE
        sbf_size = Config.SBF_SIZE
        num_mp_layers = Config.NUM_MP_LAYERS
        num_trans_layers = Config.NUM_TRANSFORMER_LAYERS
        num_heads = Config.NUM_ATTENTION_HEADS
        trans_dim_feedforward = Config.TRANSFORMER_DIM_FEEDFORWARD
        dropout = Config.DROPOUT

        # ---------------------------------------------------------------------
        # 1. Embeddings & Basis Functions
        # ---------------------------------------------------------------------
        # Atom Embedding: H, C, N, O, F (Max atomic num ~9, using 20 for safety)
        self.atom_embedding = nn.Embedding(20, hidden_dim)

        # Coupling Type Embedding
        self.type_embedding = nn.Embedding(len(Config.COUPLING_TYPES), hidden_dim)

        # Geometric Basis Functions
        # RBF for edge distances
        self.rbf_expansion = RBFExpansion(
            start=0.0, end=Config.CUTOFF_RADIUS, num_centers=rbf_size
        )
        # SBF for triplet angles
        self.sbf_expansion = SBFExpansion(num_centers=sbf_size)

        # Edge Initialization Projector
        # Projects RBF features to hidden dimension
        self.edge_embedding = nn.Linear(rbf_size, hidden_dim)

        # ---------------------------------------------------------------------
        # 2. Local Geometric Backbone (DMPNN)
        # ---------------------------------------------------------------------
        self.mp_layers = nn.ModuleList(
            [
                DMPNNLayer(hidden_dim, rbf_size, sbf_size, dropout)
                for _ in range(num_mp_layers)
            ]
        )

        # ---------------------------------------------------------------------
        # 3. Global Interaction (Transformer)
        # ---------------------------------------------------------------------
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=trans_dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_trans_layers)
            ]
        )

        # ---------------------------------------------------------------------
        # 4. Readout Head
        # ---------------------------------------------------------------------
        # Input: Atom_u (Hidden) + Atom_v (Hidden) + Type (Hidden)
        self.readout_mlp = MLP(
            input_dim=hidden_dim * 3,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=3,
            dropout=dropout,
        )

    def forward(self, data):
        """
        Forward pass of the HGA-Net.

        Args:
            data (torch_geometric.data.Data or Batch): Batch of molecular graphs.
                Contains x, edge_index, edge_attr, batch, coupling_atom_0, etc.

        Returns:
            torch.Tensor: Predicted scalar coupling constants [Batch, 1]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # ---------------------------------------------------------------------
        # 1. Initialization
        # ---------------------------------------------------------------------
        # Embed atoms
        h_nodes = self.atom_embedding(x)  # [NumAtoms, Hidden]

        # Compute initial edge features
        # Calculate distance for RBF
        dist = torch.norm(edge_attr, dim=1)
        rbf_feat = self.rbf_expansion(dist)

        # Initialize edge hidden states
        # Combine geometric projection with source and target atom features
        # This provides the initial directional context
        row, col = edge_index
        h_edges = self.edge_embedding(rbf_feat) + h_nodes[row] + h_nodes[col]

        # ---------------------------------------------------------------------
        # 2. Local Geometric Encoding (DMPNN)
        # ---------------------------------------------------------------------
        # Iteratively update edge features based on neighbors and geometry
        for layer in self.mp_layers:
            h_edges = layer(
                x,
                edge_index,
                edge_attr,
                h_edges,
                self.rbf_expansion,
                self.sbf_expansion,
            )

        # ---------------------------------------------------------------------
        # 3. Aggregation (Edge -> Node)
        # ---------------------------------------------------------------------
        # Aggregate incoming edge messages to update node features
        # scatter_sum sums features from edges pointing to the same node (col)
        m_nodes = scatter_sum(h_edges, col, dim=0, dim_size=x.size(0))

        # Combine aggregated messages with original atom embeddings
        h_nodes = h_nodes + m_nodes

        # ---------------------------------------------------------------------
        # 4. Global Interaction (Transformer)
        # ---------------------------------------------------------------------
        if len(self.transformer_layers) > 0:
            # Convert sparse graph batch to dense batch for Transformer
            # h_dense: [BatchSize, MaxNodes, Hidden]
            # mask: [BatchSize, MaxNodes] (True for real nodes, False for padding)
            h_dense, mask = to_dense_batch(h_nodes, batch)

            # Transformer expects src_key_padding_mask where True indicates padding
            padding_mask = ~mask

            for layer in self.transformer_layers:
                h_dense = layer(h_dense, src_key_padding_mask=padding_mask)

            # Flatten back to sparse format [NumAtoms, Hidden]
            h_nodes = h_dense[mask]

        # ---------------------------------------------------------------------
        # 5. Readout
        # ---------------------------------------------------------------------
        # Extract features for the specific atom pairs involved in coupling
        idx0 = data.coupling_atom_0.squeeze()
        idx1 = data.coupling_atom_1.squeeze()
        type_idx = data.coupling_type.squeeze()

        feat0 = h_nodes[idx0]
        feat1 = h_nodes[idx1]
        feat_type = self.type_embedding(type_idx)

        # Concatenate features: [u, v, type]
        pair_features = torch.cat([feat0, feat1, feat_type], dim=-1)

        # Predict coupling constant
        out = self.readout_mlp(pair_features)

        return out
