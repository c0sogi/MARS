import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from library.config import ModelConfig


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class InteractionBlock(nn.Module):
    """
    Directional Message Passing Layer.
    Updates edge embeddings based on neighboring edges (triplets) and geometric features (SBF).
    """

    def __init__(self, hidden_dim, num_rbf, num_sbf):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transformations for incoming messages and geometric features
        self.dense_kj = nn.Linear(hidden_dim, hidden_dim)
        self.dense_sbf = nn.Linear(num_rbf * num_sbf, hidden_dim)

        # Update function (Residual MLP)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
        )

    def forward(self, x_edge, sbf, triplet_indices, num_edges):
        """
        Args:
            x_edge: Edge embeddings [E, hidden_dim]
            sbf: Spherical Basis Features for triplets [T, num_sbf * num_rbf]
            triplet_indices: Indices [2, T] where row 0 is k->j and row 1 is j->i
            num_edges: Total number of edges E
        """
        # Unpack indices for triplets (k->j, j->i)
        idx_kj, idx_ji = triplet_indices[0], triplet_indices[1]

        # 1. Transform incoming edge features (k->j)
        # Handle case where triplet_indices might be empty (e.g., diatomic molecules)
        if sbf.size(0) > 0:
            h_kj = self.dense_kj(x_edge)  # [E, D]
            h_kj = h_kj[idx_kj]  # [T, D]

            # 2. Transform geometric features
            w_sbf = self.dense_sbf(sbf)  # [T, D]

            # 3. Interaction (Hadamard product)
            msg = h_kj * w_sbf  # [T, D]

            # 4. Aggregate messages to target edge (j->i)
            agg = scatter_add(msg, idx_ji, dim=0, dim_size=num_edges)  # [E, D]
        else:
            agg = torch.zeros((num_edges, self.hidden_dim), device=x_edge.device)

        # 5. State Update with Residual Connection
        # Concatenate current state with aggregated messages
        combined = torch.cat([x_edge, agg], dim=-1)
        update = self.update_mlp(combined)

        return x_edge + update


class GeometricEncoder(nn.Module):
    """
    Backbone Encoder using stacked InteractionBlocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Initial Embeddings
        self.atom_emb = nn.Embedding(config.num_atom_types, config.hidden_dim)
        self.rbf_lin = nn.Linear(config.num_rbf, config.hidden_dim)

        # Initialization MLP to create first edge state from nodes and RBF
        self.init_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            Swish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Stack of Message Passing Layers
        self.layers = nn.ModuleList(
            [
                InteractionBlock(config.hidden_dim, config.num_rbf, config.num_sbf)
                for _ in range(config.num_mp_layers)
            ]
        )

    def forward(self, z, edge_index, edge_rbf, triplet_indices, triplet_sbf):
        """
        Args:
            z: Atom types [N]
            edge_index: Graph connectivity [2, E]
            edge_rbf: Radial Basis Features [E, num_rbf]
            triplet_indices: [2, T]
            triplet_sbf: [T, SBF_dim]
        Returns:
            x_edge: Final edge embeddings [E, D]
            h_nodes_init: Initial node embeddings [N, D] (for skip connection)
        """
        row, col = edge_index

        # Initialize Node and Edge Features
        h_nodes_init = self.atom_emb(z)  # [N, D]
        h_u = h_nodes_init[row]  # [E, D]
        h_v = h_nodes_init[col]  # [E, D]
        h_rbf = self.rbf_lin(edge_rbf)  # [E, D]

        # Create initial edge state: m_ji = MLP(h_j || h_i || rbf_ji)
        x_edge = torch.cat([h_u, h_v, h_rbf], dim=-1)
        x_edge = self.init_mlp(x_edge)  # [E, D]

        num_edges = x_edge.size(0)

        # Apply Message Passing Layers
        for layer in self.layers:
            x_edge = layer(x_edge, triplet_sbf, triplet_indices, num_edges)

        return x_edge, h_nodes_init


class GlobalInteraction(nn.Module):
    """
    Global Interaction Module using a Graph Transformer.
    Aggregates edge embeddings to nodes and applies Self-Attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Projection to aggregate edge embeddings to nodes
        self.edge_to_node = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), Swish()
        )

        # Transformer Encoder
        # batch_first=True expects [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_transformer_layers
        )

    def forward(self, x_edge, x_initial, edge_index, batch, num_nodes):
        """
        Args:
            x_edge: Contextualized edge embeddings [E, D]
            x_initial: Initial node embeddings [N, D] (Skip connection)
            edge_index: [2, E]
            batch: Batch index for nodes [N]
            num_nodes: Total nodes N
        """
        # 1. Aggregate edge messages to target nodes
        # edge_index[1] is the target node index
        tgt_idx = edge_index[1]
        edge_msg = self.edge_to_node(x_edge)

        # Sum incoming edge features to get node features
        x_node = scatter_add(edge_msg, tgt_idx, dim=0, dim_size=num_nodes)  # [N, D]

        # 2. Add residual connection from initial node embedding
        x_node = x_node + x_initial

        # 3. Prepare for Transformer (Dense Batching)
        # Convert disjoint graph [N, D] to dense batch [B, MaxNodes, D]
        # mask is [B, MaxNodes] where True indicates valid node
        x_dense, mask = to_dense_batch(x_node, batch)

        # Transformer expects src_key_padding_mask where True indicates PADDING (ignore)
        # to_dense_batch mask has True for VALID nodes. So we invert it.
        padding_mask = ~mask

        # 4. Apply Transformer
        x_trans = self.transformer(
            x_dense, src_key_padding_mask=padding_mask
        )  # [B, MaxNodes, D]

        # 5. Flatten back to disjoint graph format [N, D]
        # Extract only valid nodes
        x_final = x_trans[mask]

        return x_final


class HybridModel(nn.Module):
    """
    Hybrid Geometric-Attention Network.
    Combines DMPNN backbone with Global Transformer and Pairwise Readout.
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # Components
        self.encoder = GeometricEncoder(config)
        self.global_interaction = GlobalInteraction(config)

        # Coupling Type Embedding
        self.type_emb = nn.Embedding(len(config.coupling_types), config.hidden_dim)

        # Pairwise Readout MLP
        # Input: Atom 0 + Atom 1 + Type Embedding
        self.readout_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            Swish(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            Swish(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, data):
        """
        Args:
            data: Dictionary containing graph batch from MoleculeDataset.
        Returns:
            pred: Scalar coupling constants [num_couplings]
        """
        # Unpack Data
        z = data["z"]
        edge_index = data["edge_index"]
        edge_rbf = data["edge_rbf"]
        triplet_indices = data["triplet_indices"]
        triplet_sbf = data["triplet_sbf"]
        batch = data["batch"]

        # 1. Geometric Encoding (Local Structure)
        x_edge, x_init = self.encoder(
            z, edge_index, edge_rbf, triplet_indices, triplet_sbf
        )

        # 2. Global Interaction (Long-range dependencies)
        num_nodes = z.size(0)
        x_node = self.global_interaction(x_edge, x_init, edge_index, batch, num_nodes)

        # 3. Pairwise Readout
        c_atom0 = data["coupling_atom0"]
        c_atom1 = data["coupling_atom1"]
        c_type = data["coupling_type"]

        # Gather node features for the coupling pairs
        h0 = x_node[c_atom0]
        h1 = x_node[c_atom1]
        h_type = self.type_emb(c_type)

        # Construct pair representation
        pair_rep = torch.cat([h0, h1, h_type], dim=-1)

        # Predict
        pred = self.readout_mlp(pair_rep)

        return pred.squeeze(-1)
