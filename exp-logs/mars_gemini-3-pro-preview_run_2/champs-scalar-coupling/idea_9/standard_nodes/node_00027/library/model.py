import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from library.config import Config


class DMPNNLayer(nn.Module):
    """
    A single layer of Directional Message Passing.
    Updates edge embeddings based on messages from incoming edges, modulated by angular (SBF) features.
    """

    def __init__(self, hidden_dim, num_sbf, num_rbf, activation="swish"):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Interaction block: Combines incoming edge features with geometric triplet features
        # SBF features are flattened (num_rbf * num_sbf)
        sbf_dim = num_rbf * num_sbf
        self.lin_sbf = nn.Linear(sbf_dim, hidden_dim, bias=False)
        self.lin_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Update function: Process aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU() if activation == "swish" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU() if activation == "swish" else nn.ReLU(),
        )

    def forward(self, x_edges, triplet_attr, triplet_index):
        """
        Args:
            x_edges: Current edge embeddings (E, hidden_dim)
            triplet_attr: SBF features for triplets (T, sbf_dim)
            triplet_index: Indices mapping triplets to edges (2, T).
                           row 0: index of incoming edge (k->j)
                           row 1: index of outgoing edge (j->i)
        """
        # 1. Prepare messages from incoming edges (k->j)
        # Get features of incoming edges involved in triplets
        idx_incoming = triplet_index[0]
        h_incoming = x_edges[idx_incoming]  # (T, hidden_dim)

        # 2. Interaction: Modulate by geometry
        # We use a Hadamard product interaction after projection
        # m_kji = Linear(h_kj) * Linear(sbf_kji)
        geom_feat = self.lin_sbf(triplet_attr)  # (T, hidden_dim)
        edge_feat = self.lin_edge(h_incoming)  # (T, hidden_dim)
        message = edge_feat * geom_feat  # (T, hidden_dim)

        # 3. Aggregate messages to outgoing edges (j->i)
        idx_outgoing = triplet_index[1]
        # Sum messages destined for the same edge index
        # Result shape: (E, hidden_dim)
        aggr_messages = scatter_sum(
            message, idx_outgoing, dim=0, dim_size=x_edges.size(0)
        )

        # 4. Update edge states
        # Concatenate current state with aggregated message
        out = torch.cat([x_edges, aggr_messages], dim=1)
        x_edges_new = self.update_mlp(out)

        # Residual connection
        return x_edges + x_edges_new


class DMPNNEncoder(nn.Module):
    """
    Backbone encoder using Directional Message Passing.
    """

    def __init__(self, hidden_dim, num_layers, num_rbf, num_sbf):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Atom Embedding
        self.atom_emb = nn.Embedding(Config.NUM_ATOM_TYPES, hidden_dim)

        # Initial Edge Embedding: cat(u, v, rbf) -> hidden
        self.edge_init = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Interaction Layers
        self.layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, num_sbf, num_rbf) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr, triplet_index, triplet_attr):
        # 1. Initialize Edge Features
        # x: (N_atoms) atom types
        src, dst = edge_index

        u_emb = self.atom_emb(x[src])
        v_emb = self.atom_emb(x[dst])

        # Concatenate atom features and RBF distance features
        edge_input = torch.cat([u_emb, v_emb, edge_attr], dim=1)
        h_edges = self.edge_init(edge_input)  # (E, hidden_dim)

        # Save initial embedding for fallback
        h_edges_0 = h_edges

        # 2. Message Passing
        for layer in self.layers:
            h_edges = layer(h_edges, triplet_attr, triplet_index)

        return h_edges, h_edges_0


class GlobalTransformer(nn.Module):
    """
    Global interaction module using Transformer Encoder.
    Aggregates local edge features to nodes, then applies self-attention.
    """

    def __init__(self, hidden_dim, num_heads, num_layers=2):
        super().__init__()

        # Transition from edges to nodes
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,  # No dropout as per requirements
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, h_edges, edge_index, num_nodes, batch):
        # 1. Aggregate edges to nodes (incoming edges)
        # h_edges: (E, hidden)
        # edge_index[1] is the target node of the directed edge
        dst = edge_index[1]

        # Sum edge features into node features
        h_nodes = scatter_sum(h_edges, dst, dim=0, dim_size=num_nodes)
        h_nodes = self.node_proj(h_nodes)  # (N, hidden)

        # 2. Prepare for Transformer (Batching)
        # PyG batches graphs into a single large graph.
        # Transformer expects (Batch, SeqLen, Dim) with padding mask,
        # OR we can treat the whole super-graph as one sequence with a custom mask.
        # However, standard Transformer attention is O(N^2).
        # Since molecules are small (<30 atoms), we can unbatch to dense or use a mask.
        # Given the varying sizes, unbatching to dense (B, MaxNodes, D) is easiest.

        # Count nodes per graph
        batch_size = batch.max().item() + 1

        # We'll use torch_geometric.utils.to_dense_batch
        from torch_geometric.utils import to_dense_batch

        x_dense, mask = to_dense_batch(h_nodes, batch)
        # x_dense: (B, max_nodes, hidden)
        # mask: (B, max_nodes)

        # Pass through Transformer
        # src_key_padding_mask: True for padded positions
        out_dense = self.transformer(x_dense, src_key_padding_mask=~mask)

        # Recover sparse format
        # Flatten out_dense based on mask
        h_nodes_global = out_dense[mask]

        return h_nodes_global


class HybridReadout(nn.Module):
    """
    Fuses Global Node embeddings and Local Edge embeddings to predict coupling.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.type_emb = nn.Embedding(Config.NUM_COUPLING_TYPES, hidden_dim)

        # Input: Node_u + Node_v + Edge_uv + Type
        in_dim = hidden_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_nodes,
        h_edges,
        h_edges_0,
        edge_index,
        target_edge_index,
        target_type,
        num_nodes,
    ):
        """
        Args:
            h_nodes: Global node embeddings (N, hidden)
            h_edges: Refined edge embeddings from DMPNN (E, hidden)
            h_edges_0: Initial edge embeddings (E, hidden) - used for fallback
            edge_index: Graph connectivity (2, E)
            target_edge_index: Target pairs (2, n_targets)
            target_type: Coupling types (n_targets,)
        """
        # 1. Gather Node Features
        u_idx = target_edge_index[0]
        v_idx = target_edge_index[1]

        h_u = h_nodes[u_idx]
        h_v = h_nodes[v_idx]

        # 2. Gather Edge Features
        # We need to map target pairs (u, v) to indices in h_edges.
        # Create a unique hash for edges to perform lookup.
        # Assumption: num_nodes in batch < 2^32.
        # To be safe across batches, we use the sparse index logic.

        # Hash function: u * large_num + v
        # We use the max node index in the batch to ensure uniqueness
        scale = num_nodes + 1

        # Graph edges hash
        graph_edge_hash = edge_index[0] * scale + edge_index[1]

        # Target edges hash
        target_edge_hash = u_idx * scale + v_idx

        # We need to find where target_edge_hash occurs in graph_edge_hash.
        # Since we are on GPU, we can't easily use a dictionary.
        # We can use searchsorted if we sort graph_edge_hash.

        sorted_hash, sort_idx = torch.sort(graph_edge_hash)

        # Find insertion points
        idx_in_sorted = torch.searchsorted(sorted_hash, target_edge_hash)

        # Clamp to valid range
        idx_in_sorted = torch.clamp(idx_in_sorted, max=len(sorted_hash) - 1)

        # Check if found
        found_hash = sorted_hash[idx_in_sorted]
        is_found = found_hash == target_edge_hash

        # Map back to original edge indices
        original_edge_indices = sort_idx[idx_in_sorted]

        # Gather features
        # If found, use h_edges (refined). If not, we have a problem because we don't have h_edges_0 for non-existent edges.
        # However, h_edges_0 was computed on edge_index.
        # If the edge is NOT in edge_index, we don't have a precomputed embedding.
        # In this case, we will use a Zero vector for the edge part (or rely on node parts).
        # Given the 5.0A cutoff, most should be found.

        batch_size_targets = target_edge_index.size(1)
        h_edge_target = torch.zeros(
            batch_size_targets,
            h_edges.size(1),
            device=h_edges.device,
            dtype=h_edges.dtype,
        )

        # Fill found edges
        if is_found.any():
            found_indices = original_edge_indices[is_found]
            h_edge_target[is_found] = h_edges[found_indices]

        # 3. Type Embedding
        h_type = self.type_emb(target_type)

        # 4. Fuse and Predict
        cat_feat = torch.cat([h_u, h_v, h_edge_target, h_type], dim=1)
        pred = self.mlp(cat_feat)

        return pred.squeeze(-1)


class HGANet(nn.Module):
    """
    Scaled Hybrid Geometric-Attention Network.
    """

    def __init__(self):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM

        # 1. Backbone
        self.encoder = DMPNNEncoder(
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            num_rbf=Config.NUM_RBF,
            num_sbf=Config.NUM_SBF,
        )

        # 2. Global Interaction
        self.transformer = GlobalTransformer(
            hidden_dim=Config.HIDDEN_DIM, num_heads=Config.NUM_HEADS
        )

        # 3. Readout
        self.readout = HybridReadout(hidden_dim=Config.HIDDEN_DIM)

    def forward(self, data):
        """
        Args:
            data: PyG Data object
        """
        # Unpack Data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        triplet_index = data.triplet_index
        triplet_attr = data.triplet_attr
        batch = data.batch

        # 1. Local Geometric Encoding
        h_edges, h_edges_0 = self.encoder(
            x, edge_index, edge_attr, triplet_index, triplet_attr
        )

        # 2. Global Interaction
        num_nodes = x.size(0)
        h_nodes = self.transformer(h_edges, edge_index, num_nodes, batch)

        # 3. Readout
        # Predict for specific target pairs
        pred = self.readout(
            h_nodes=h_nodes,
            h_edges=h_edges,
            h_edges_0=h_edges_0,
            edge_index=edge_index,
            target_edge_index=data.target_edge_index,
            target_type=data.target_type,
            num_nodes=num_nodes,
        )

        return pred
