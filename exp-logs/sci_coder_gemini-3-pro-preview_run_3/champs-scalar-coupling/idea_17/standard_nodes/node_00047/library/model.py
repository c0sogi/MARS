import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from library.config import Config


class ShiftedSoftplus(nn.Module):
    """
    Shifted Softplus activation function: f(x) = ln(0.5 * e^x + 0.5)
    Used to ensure the activation is close to linear for small inputs and smooth everywhere.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class RBFExpansion(nn.Module):
    """
    Expands a scalar distance into a vector of Radial Basis Functions (Gaussians).
    """

    def __init__(self, start=0.0, stop=Config.CUTOFF, n_gaussians=Config.N_RBF):
        super(RBFExpansion, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        # Width of the Gaussians
        width = (stop - start) / (n_gaussians - 1)
        self.coeff = -0.5 / (width**2)
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (E,) or (E, 1) containing distances.
        Returns:
            Tensor of shape (E, n_gaussians)
        """
        # (E, 1) - (1, n_gaussians) -> (E, n_gaussians)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    Continuous Filter Convolution Block.
    Updates node embeddings based on interacting neighbors and continuous edge filters.
    """

    def __init__(self, hidden_channels, n_rbf, n_filters):
        super(InteractionBlock, self).__init__()

        # Filter Generator: Transforms RBF edge features into spatial filters
        self.mlp = nn.Sequential(
            nn.Linear(n_rbf, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters),
        )

        # Transformation for neighbor features
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        # Update network for aggregated messages
        self.update_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, hidden_channels)
            edge_index: Edge indices (2, E)
            edge_attr: RBF expanded edge features (E, n_rbf)
        """
        # 1. Generate continuous filters from edge distances
        W = self.mlp(edge_attr)  # (E, n_filters)

        # 2. Transform neighbor node features
        row, col = edge_index
        x_j = self.lin(x[col])  # (E, hidden_channels)

        # 3. Continuous Filter Convolution (Element-wise interaction)
        m = x_j * W

        # 4. Aggregate messages (Sum pooling)
        # scatter_add(src, index, dim, dim_size)
        aggr = scatter(m, row, dim=0, dim_size=x.size(0), reduce="add")

        # 5. Update node embeddings with residual connection
        x = x + self.update_lin(aggr)

        return x


class InteractionAwareHead(nn.Module):
    """
    Readout head that predicts coupling constants.
    explicitly models the interaction between the two atoms in the pair.
    """

    def __init__(self, hidden_channels, n_rbf, num_coupling_types, hidden_dims):
        super(InteractionAwareHead, self).__init__()

        self.type_embedding = nn.Embedding(num_coupling_types, hidden_channels)

        # Input Dimension Construction:
        # h_i (hidden) + h_j (hidden) + dot_prod (1) + rbf_dist (n_rbf) + type_emb (hidden)
        input_dim = (hidden_channels * 2) + 1 + n_rbf + hidden_channels

        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.SiLU())  # Modern activation for the readout
            curr_dim = h_dim

        # Final projection to scalar
        layers.append(nn.Linear(curr_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, coupling_index, coupling_type, coupling_dist_rbf):
        """
        Args:
            x: Node embeddings (N, hidden_channels)
            coupling_index: Indices of coupling pairs (2, C)
            coupling_type: Coupling type indices (C,)
            coupling_dist_rbf: RBF features of the distance between coupling pairs (C, n_rbf)
        """
        idx_i, idx_j = coupling_index

        # Gather node embeddings
        h_i = x[idx_i]  # (C, hidden_channels)
        h_j = x[idx_j]  # (C, hidden_channels)

        # Explicit Multiplicative Interaction (Dot Product)
        # Models the alignment/interaction strength between atom states
        dot_prod = (h_i * h_j).sum(dim=1, keepdim=True)  # (C, 1)

        # Coupling Type Embedding
        type_emb = self.type_embedding(coupling_type)  # (C, hidden_channels)

        # Concatenate all features
        # [h_i, h_j, h_i . h_j, dist_ij, type_emb]
        out = torch.cat([h_i, h_j, dot_prod, coupling_dist_rbf, type_emb], dim=1)

        # Predict
        return self.mlp(out)


class MPDCFN(nn.Module):
    """
    Molecule-Parallel Deep Continuous Filter Network.
    """

    def __init__(self):
        super(MPDCFN, self).__init__()

        # Hyperparameters from Config
        self.num_atom_types = Config.NUM_ATOM_TYPES
        self.num_coupling_types = Config.NUM_COUPLING_TYPES
        self.n_atom_basis = Config.N_ATOM_BASIS
        self.n_filters = Config.N_FILTERS
        self.n_interactions = Config.N_INTERACTIONS
        self.n_rbf = Config.N_RBF
        self.cutoff = Config.CUTOFF
        self.readout_dims = Config.READOUT_HIDDEN_DIMS

        # 1. Atom Embedding
        self.embedding = nn.Embedding(self.num_atom_types, self.n_atom_basis)

        # 2. RBF Expansion Layer
        self.rbf = RBFExpansion(start=0.0, stop=self.cutoff, n_gaussians=self.n_rbf)

        # 3. Interaction Blocks (Backbone)
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(self.n_atom_basis, self.n_rbf, self.n_filters)
                for _ in range(self.n_interactions)
            ]
        )

        # 4. Readout Head
        self.head = InteractionAwareHead(
            self.n_atom_basis, self.n_rbf, self.num_coupling_types, self.readout_dims
        )

    def forward(self, data):
        """
        Args:
            data: Dictionary or PyG-like object containing:
                - x: Atom types (N,)
                - pos: Atom coordinates (N, 3)
                - edge_index: Radius graph edge indices (2, E)
                - edge_attr: Radius graph edge distances (E, 1)
                - coupling_index: Target coupling pairs (2, C)
                - coupling_type: Target coupling types (C,)
        Returns:
            Tensor of shape (C,) containing predicted standardized coupling constants.
        """
        x = data["x"]
        pos = data["pos"]
        edge_index = data["edge_index"]
        edge_dist = data["edge_attr"]

        coupling_index = data["coupling_index"]
        coupling_type = data["coupling_type"]

        # --- 1. Initial Embedding ---
        h = self.embedding(x)  # (N, n_atom_basis)

        # --- 2. Precompute Edge Features ---
        # Expand edge distances for the message passing graph
        if edge_dist.numel() > 0:
            edge_rbf = self.rbf(edge_dist.squeeze(-1))  # (E, n_rbf)
        else:
            edge_rbf = torch.empty((0, self.n_rbf), device=x.device)

        # --- 3. Message Passing Backbone ---
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_rbf)

        # --- 4. Coupling-Specific Feature Engineering ---
        # We need the exact distance between the specific coupling pairs.
        # While these edges likely exist in edge_index, extracting them by index matching
        # is slow. It's faster to recompute the distance for the requested pairs.
        c_i, c_j = coupling_index
        vec_diff = pos[c_i] - pos[c_j]
        c_dist = vec_diff.norm(dim=1)  # (C,)

        # Expand coupling distances
        c_rbf = self.rbf(c_dist)  # (C, n_rbf)

        # --- 5. Readout ---
        out = self.head(h, coupling_index, coupling_type, c_rbf)

        return out.squeeze(-1)
