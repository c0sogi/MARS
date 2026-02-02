import torch
import torch.nn as nn
from torch_scatter import scatter
from library.config import Config
from library.layers import InteractionBlock, GlobalAttentionBlock, ReadoutBlock


class HGANet(nn.Module):
    """
    Hybrid Geometric-Attention Network (HGA-Net).

    Architecture:
    1. Local Geometric Encoder: Directional Message Passing (DMPNN) with RBF/SBF.
    2. Global Interaction: Graph Transformer for long-range dependencies.
    3. Deterministic Readout: Fuses Global Node + Local Edge features.
    """

    def __init__(self):
        super(HGANet, self).__init__()

        # ---------------------------------------------------------------------
        # Hyperparameters
        # ---------------------------------------------------------------------
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_rbf = Config.NUM_RBF
        self.num_sbf = Config.NUM_SBF
        self.num_mp_layers = Config.NUM_MP_LAYERS

        # ---------------------------------------------------------------------
        # 1. Embeddings & Initialization
        # ---------------------------------------------------------------------
        # Atom Embedding: Maps atom types (0-5) to hidden_dim
        # Map size is small (H, C, N, O, F + unknown), so 6 is sufficient.
        self.atom_embedding = nn.Embedding(6, self.hidden_dim)

        # Project RBF distance features to hidden dimension for edge initialization
        self.rbf_proj = nn.Linear(self.num_rbf, self.hidden_dim, bias=False)

        # ---------------------------------------------------------------------
        # 2. Local Geometric Encoder (Backbone)
        # ---------------------------------------------------------------------
        # Stack of Directional Message Passing layers
        self.mp_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=self.hidden_dim,
                    num_rbf=self.num_rbf,
                    num_sbf=self.num_sbf,
                )
                for _ in range(self.num_mp_layers)
            ]
        )

        # Projection to merge aggregated edge messages into node features
        self.node_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # ---------------------------------------------------------------------
        # 3. Global Interaction Module
        # ---------------------------------------------------------------------
        # Stack of Graph Transformer layers
        self.global_layers = nn.ModuleList(
            [
                GlobalAttentionBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=Config.TRANSFORMER_HEADS,
                    ff_dim=Config.TRANSFORMER_FF_DIM,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.TRANSFORMER_LAYERS)
            ]
        )

        # ---------------------------------------------------------------------
        # 4. Readout
        # ---------------------------------------------------------------------
        # Fuses features and predicts scalar coupling constant
        self.readout = ReadoutBlock(hidden_dim=self.hidden_dim)

    def forward(self, data):
        """
        Forward pass of the HGA-Net.

        Args:
            data (dict): Batch dictionary containing graph features and coupling targets.
                         Keys: atom_types, edge_index, edge_rbf, triplet_indices,
                               triplet_sbf, batch, coupling_atom_0, coupling_atom_1,
                               coupling_types.

        Returns:
            Tensor: Predicted scalar coupling constants (N_couplings, 1).
        """
        # Unpack data
        atom_types = data["atom_types"]  # (N_nodes,)
        edge_index = data["edge_index"]  # (2, N_edges)
        edge_rbf = data["edge_rbf"]  # (N_edges, num_rbf)
        triplet_indices = data["triplet_indices"]  # (2, N_triplets)
        triplet_sbf = data["triplet_sbf"]  # (N_triplets, num_sbf)
        batch = data["batch"]  # (N_nodes,)

        # ---------------------------------------------------------------------
        # Phase 1: Initialization
        # ---------------------------------------------------------------------
        # Embed atoms
        x = self.atom_embedding(atom_types)  # (N_nodes, hidden_dim)

        # Initialize Edge Embeddings
        # h_edge = x_src + x_dst + proj(rbf)
        src, dst = edge_index
        rbf_emb = self.rbf_proj(edge_rbf)
        h_edge = x[src] + x[dst] + rbf_emb  # (N_edges, hidden_dim)

        # ---------------------------------------------------------------------
        # Phase 2: Local Geometric Encoding (Message Passing)
        # ---------------------------------------------------------------------
        # Update edge embeddings based on geometric interactions
        for block in self.mp_blocks:
            h_edge = block(h_edge, edge_rbf, triplet_sbf, triplet_indices)

        # ---------------------------------------------------------------------
        # Phase 3: Edge -> Node Aggregation
        # ---------------------------------------------------------------------
        # Aggregate incoming edge messages to update node states
        # dst indices represent the target of the directed edge
        m_agg = scatter(h_edge, dst, dim=0, dim_size=x.size(0), reduce="sum")

        # Residual update of node features
        x_local = x + self.node_proj(m_agg)

        # ---------------------------------------------------------------------
        # Phase 4: Global Interaction (Transformer)
        # ---------------------------------------------------------------------
        # Apply Global Attention to capture long-range dependencies
        # Iterate through layers (cannot use Sequential due to 'batch' arg)
        x_global = x_local
        for layer in self.global_layers:
            x_global = layer(x_global, batch)

        # ---------------------------------------------------------------------
        # Phase 5: Feature Extraction & Readout
        # ---------------------------------------------------------------------
        c_atom_0 = data["coupling_atom_0"]
        c_atom_1 = data["coupling_atom_1"]
        c_types = data["coupling_types"]

        # 1. Get Global Node Embeddings for the pair
        node_u = x_global[c_atom_0]
        node_v = x_global[c_atom_1]

        # 2. Get Local Edge Embeddings for the pair (u -> v)
        # We need to efficiently find the index of the edge connecting u->v in the sparse graph.

        num_nodes = x.size(0)

        # Create unique 1D hash for edges in the graph: src * N + dst
        # Using int64 to prevent overflow (N can be ~60k in a batch)
        edge_hash = edge_index[0] * num_nodes + edge_index[1]

        # Create hash for the query coupling pairs
        query_hash = c_atom_0 * num_nodes + c_atom_1

        # Sort graph edges to enable binary search (searchsorted)
        sorted_edge_hash, perm = torch.sort(edge_hash)

        # Find indices of query pairs in the sorted edge list
        idx = torch.searchsorted(sorted_edge_hash, query_hash)

        # Clamp indices to be within valid range (handles cases where edge might be missing,
        # though data gen ensures coverage)
        idx = idx.clamp(max=len(edge_hash) - 1)

        # Map back to original unsorted indices
        real_idx = perm[idx]

        # Retrieve the edge embeddings
        edge_uv = h_edge[real_idx]

        # Safety Mask: Zero out embedding if the edge wasn't actually found
        # (i.e., if hash doesn't match, searchsorted returned insertion point)
        found_mask = (edge_hash[real_idx] == query_hash).unsqueeze(-1)
        edge_uv = edge_uv * found_mask.float()

        # 3. Predict
        pred = self.readout(node_u, node_v, edge_uv, c_types)

        return pred
