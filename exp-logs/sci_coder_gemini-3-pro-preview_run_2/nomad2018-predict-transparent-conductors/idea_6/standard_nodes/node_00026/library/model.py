import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from library.config import Config


class RBFExpansion(nn.Module):
    """
    Expands a distance tensor into a vector of radial basis functions (Gaussian).
    """

    def __init__(self, vmin=0, vmax=8.0, bins=40):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        # Register centers as a buffer so it's part of the state_dict but not a parameter
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        # Calculate gamma based on bin width
        self.gamma = 1.0 / ((vmax - vmin) / bins) ** 2

    def forward(self, distance):
        """
        Args:
            distance: Tensor of shape (E, 1) or (E,)
        Returns:
            Tensor of shape (E, bins)
        """
        # Ensure distance is (E, 1) for broadcasting
        if distance.dim() == 1:
            distance = distance.unsqueeze(1)

        # (E, 1) - (bins,) -> (E, bins)
        base = distance - self.centers
        return torch.exp(-self.gamma * (base**2))


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network Layer.
    Implements the update rule:
    h_i' = h_i + Sum_j [ sigmoid(z_ij W_f + b_f) * softplus(z_ij W_s + b_s) ]
    where z_ij = [h_i, h_j, e_ij]
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")  # Sum aggregation
        # Cite debug_lesson_7: Avoid overwriting reserved 'node_dim' attribute
        self.emb_dim = node_dim
        self.edge_dim = edge_dim

        # Input dimension: Source Node + Target Node + Edge Attribute
        input_dim = 2 * node_dim + edge_dim

        self.lin_f = nn.Linear(input_dim, node_dim)
        self.lin_s = nn.Linear(input_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: (N, node_dim)
        # edge_index: (2, E)
        # edge_attr: (E, edge_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Calculate filter (gate) and core (signal)
        gate = torch.sigmoid(self.lin_f(z))
        core = F.softplus(self.lin_s(z))

        return gate * core

    def update(self, aggr_out, x):
        # Residual connection and batch norm
        return self.bn(x + aggr_out)


class VirtualNodeInteraction(nn.Module):
    """
    Handles interaction between the atomic graph and the global virtual node.
    1. Aggregates atom info to update virtual node.
    2. Broadcasts virtual node info back to atoms.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        # MLP to update virtual node state based on aggregated atom features
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, virtual_node, batch):
        """
        Args:
            x: Atom features (N, hidden_dim)
            virtual_node: Virtual node features (batch_size, hidden_dim)
            batch: Batch indices for atoms (N,)
        """
        # 1. Atom-to-Virtual Aggregation
        # Sum pool atom features for each graph in the batch
        x_aggr = global_add_pool(x, batch)  # (batch_size, hidden_dim)

        # 2. Update Virtual Node
        # Concatenate current virtual state with aggregated atom state
        v_in = torch.cat([virtual_node, x_aggr], dim=1)

        # Residual update for virtual node
        v_update = self.mlp(v_in)
        virtual_node_new = virtual_node + v_update

        # 3. Virtual-to-Atom Broadcast
        # Add updated virtual node features back to atoms (broadcast via batch index)
        # virtual_node_new[batch] expands (batch_size, hidden) -> (N, hidden)
        x_new = x + virtual_node_new[batch]

        return x_new, virtual_node_new


class VNCGCNN(nn.Module):
    """
    Virtual-Node Augmented Crystal Graph Convolutional Network.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.atom_embedding_dim = config.ATOM_EMBEDDING_DIM
        self.n_rbf = config.N_RBF
        self.cutoff = config.CUTOFF_RADIUS

        # 1. Embeddings
        # Atomic number embedding (1-118 safe range, using 120)
        self.embedding = nn.Embedding(120, self.atom_embedding_dim)

        # Edge RBF expansion
        self.rbf = RBFExpansion(vmin=0, vmax=self.cutoff, bins=self.n_rbf)

        # Virtual Node Initialization MLP
        # Maps standardized global features (10) to hidden dim
        self.virtual_init = nn.Sequential(
            nn.Linear(config.N_GLOBAL_FEATURES, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 2. Interaction Blocks
        self.convs = nn.ModuleList()
        self.vn_interactions = nn.ModuleList()

        for _ in range(config.N_CONV_LAYERS):
            self.convs.append(CGCNNConv(self.hidden_dim, self.n_rbf))
            self.vn_interactions.append(
                VirtualNodeInteraction(self.hidden_dim, config.DROPOUT)
            )

        # 3. Readout
        # Concatenate Virtual Node + Mean Pooled Atoms -> Output
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, 2),  # 2 targets: formation energy, bandgap
        )

    def forward(self, data):
        # Unpack data
        atom_numbers = data.x
        edge_index = data.edge_index
        edge_dist = data.edge_attr
        global_feats = data.global_features
        batch = data.batch

        # Handle case where batch is None (single graph inference)
        if batch is None:
            batch = torch.zeros(
                atom_numbers.size(0), dtype=torch.long, device=atom_numbers.device
            )

        # 1. Initial Embeddings
        x = self.embedding(atom_numbers)  # (N, atom_embedding_dim)

        # Ensure hidden dim matches embedding dim (usually they are set same in config)
        # If not, a projection would be needed here. Assuming they are same for CGCNN.

        edge_attr = self.rbf(edge_dist)  # (E, n_rbf)

        # Initialize Virtual Node from global features
        # global_feats: (batch_size, 10)
        virtual_node = self.virtual_init(global_feats)  # (batch_size, hidden_dim)

        # 2. Interaction Loop
        for conv, vn_interaction in zip(self.convs, self.vn_interactions):
            # Virtual Node Interaction (Atom <-> Virtual)
            # Updates both x and virtual_node
            x, virtual_node = vn_interaction(x, virtual_node, batch)

            # CGCNN Convolution (Atom <-> Atom)
            # Updates x based on neighbors
            x = conv(x, edge_index, edge_attr)

        # 3. Readout
        # Mean pool atom features to get graph representation
        x_mean = global_mean_pool(x, batch)  # (batch_size, hidden_dim)

        # Concatenate with final virtual node state
        out_vec = torch.cat([x_mean, virtual_node], dim=1)  # (batch_size, 2*hidden_dim)

        # Final prediction
        out = self.output_mlp(out_vec)

        return out
