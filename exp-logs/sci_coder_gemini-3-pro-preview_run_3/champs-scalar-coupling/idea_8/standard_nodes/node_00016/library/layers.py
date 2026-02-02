import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from library.config import Config
from library.utils import GaussianSmearing


class CFConv(MessagePassing):
    """
    Continuous Filter Convolution Layer.
    Generates dynamic filters from edge attributes using an MLP.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__(aggr="add")
        # Filter Generator: Transforms edge attributes (RBFs + embeddings) into filter weights
        self.filter_network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
        )
        # Transformation for neighbor node features before aggregation
        self.lin_neighbor = nn.Linear(node_dim, out_dim)
        # Update network applied after aggregation
        self.lin_update = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Generate dynamic filters
        # edge_attr: [E, edge_dim] -> W: [E, out_dim]
        W = self.filter_network(edge_attr)

        # Propagate messages
        # x: [N, node_dim]
        out = self.propagate(edge_index, x=x, W=W)

        # Apply final update
        out = self.lin_update(out)
        return out

    def message(self, x_j, W):
        # x_j: [E, node_dim] (Features of neighbor nodes)
        # W: [E, out_dim] (Generated filters)
        # Element-wise multiplication (SchNet-style interaction)
        return self.lin_neighbor(x_j) * W


class InteractionBlock(nn.Module):
    """
    Standard SchNet-like Interaction Block.
    Updates atom representations based on distance-weighted neighbors.
    """

    def __init__(self, hidden_channels, num_rbf_dist):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)
        self.conv = CFConv(
            node_dim=hidden_channels,
            edge_dim=num_rbf_dist,
            hidden_dim=hidden_channels,
            out_dim=hidden_channels,
        )

    def forward(self, x, edge_index, dist_rbf):
        h_in = self.norm(x)
        h_out = self.conv(h_in, edge_index, dist_rbf)
        return x + h_out


class FastGNN(nn.Module):
    """
    Optimized GNN for Scalar Coupling Prediction.
    Removes Line Graph complexity to enable training on full dataset.
    """

    def __init__(self):
        super().__init__()
        self.hidden_channels = Config.HIDDEN_CHANNELS
        self.num_layers = Config.NUM_LAYERS
        self.num_rbf_dist = Config.NUM_RBF_DISTANCE

        # Embeddings
        self.embedding_atom = nn.Embedding(Config.NUM_ATOM_TYPES, self.hidden_channels)

        # Edge Embedding (from Distance RBF)
        self.embedding_edge = nn.Sequential(
            nn.Linear(self.num_rbf_dist, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # RBF Expander
        self.rbf_dist = GaussianSmearing(
            start=0.0, stop=Config.CUTOFF_RADIUS, num_gaussians=self.num_rbf_dist
        )

        # Backbone
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(self.hidden_channels, self.num_rbf_dist)
                for _ in range(self.num_layers)
            ]
        )

        # Readout Heads
        # Input: Atom_i + Atom_j + Edge_ij
        readout_input_dim = 3 * self.hidden_channels

        self.heads = nn.ModuleList()
        for _ in range(Config.NUM_HEADS):
            self.heads.append(
                nn.Sequential(
                    nn.Linear(readout_input_dim, self.hidden_channels),
                    nn.SiLU(),
                    nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                    nn.SiLU(),
                    nn.Linear(self.hidden_channels // 2, 1),
                )
            )

    def get_edge_embeddings_for_couplings(
        self, edge_embeddings, edge_index, coupling_index, num_nodes
    ):
        device = edge_embeddings.device
        E = edge_index.size(1)
        indices = edge_index
        values = torch.arange(1, E + 1, device=device)

        # Sparse to Dense for lookup (efficient enough for batch size)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        adj_dense = adj.to_dense()

        rows = coupling_index[0]
        cols = coupling_index[1]
        found_indices = adj_dense[rows, cols]

        mask = found_indices > 0
        valid_indices = found_indices[mask] - 1

        K = coupling_index.size(1)
        out_embeddings = torch.zeros(K, self.hidden_channels, device=device)

        if mask.any():
            out_embeddings[mask] = edge_embeddings[valid_indices]

        return out_embeddings

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 1. Feature Expansion
        dist_rbf = self.rbf_dist(edge_attr)

        # 2. Initialization
        h_atom = self.embedding_atom(x)
        h_edge = self.embedding_edge(dist_rbf)

        # 3. Message Passing
        for block in self.blocks:
            h_atom = block(h_atom, edge_index, dist_rbf)

        # 4. Readout
        coupling_index = data.coupling_index
        coupling_type = data.coupling_type

        h_0 = h_atom[coupling_index[0]]
        h_1 = h_atom[coupling_index[1]]

        # Edge Injection (Cite Lesson 11)
        h_e_pair = self.get_edge_embeddings_for_couplings(
            h_edge, edge_index, coupling_index, data.num_nodes
        )

        z = torch.cat([h_0, h_1, h_e_pair], dim=-1)

        pred_coupling = torch.zeros(z.size(0), 1, device=z.device)
        for t_idx in range(Config.NUM_HEADS):
            mask = coupling_type == t_idx
            if mask.any():
                pred_coupling[mask] = self.heads[t_idx](z[mask])

        return pred_coupling.squeeze(-1)
