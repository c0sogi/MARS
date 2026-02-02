import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_cluster import radius_graph
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands scalar distances into a vector of Radial Basis Function (RBF) values.
    """

    def __init__(self, start=0.0, stop=5.0, n_rbf=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_rbf)
        self.coeff = -0.5 / ((stop - start) / (n_rbf - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (N_edges, ) containing distances.
        Returns:
            Tensor of shape (N_edges, n_rbf) containing RBF features.
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    Continuous Filter Convolution Block (Node-Centric).
    Updates node features based on neighbors and continuous edge filters.
    """

    def __init__(self, hidden_dim, n_rbf):
        super(InteractionBlock, self).__init__()
        self.hidden_dim = hidden_dim

        # Transformation of input node features
        self.dense_in = nn.Linear(hidden_dim, hidden_dim)

        # Filter generator: Maps RBF edge features to filter weights
        self.filter_network = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformation of aggregated features
        self.dense_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N_atoms, hidden_dim)
            edge_index: Graph connectivity (2, N_edges)
            edge_attr: Edge RBF features (N_edges, n_rbf)
        """
        row, col = edge_index

        # 1. Transform node features
        x_in = self.dense_in(x)  # (N_atoms, hidden_dim)

        # 2. Generate continuous filters from edge attributes
        W = self.filter_network(edge_attr)  # (N_edges, hidden_dim)

        # 3. Interaction: Element-wise product of neighbor features and filters
        # Gather neighbor features x_j (source nodes are in 'row' or 'col' depending on flow convention)
        # Standard convention: flow="source_to_target", usually col -> row
        x_j = x_in[col]
        m = x_j * W  # (N_edges, hidden_dim)

        # 4. Aggregation: Sum messages to target nodes (row)
        # scatter(src, index, dim, dim_size, reduce)
        aggr = scatter(m, row, dim=0, dim_size=x.size(0), reduce="add")

        # 5. Update and Residual
        out = self.dense_out(aggr)
        return x + out


class ConditionalInteractionHead(nn.Module):
    """
    Readout module that predicts scalar coupling constants.
    Explicitly models pairwise interactions.
    """

    def __init__(self, hidden_dim, n_rbf, coupling_embed_dim, n_coupling_types):
        super(ConditionalInteractionHead, self).__init__()

        self.use_coupling_embed = Config.USE_COUPLING_EMBED

        # Coupling Type Embedding
        if self.use_coupling_embed:
            self.type_embedding = nn.Embedding(n_coupling_types, coupling_embed_dim)
            input_dim = (
                2 * hidden_dim + 1 + n_rbf + coupling_embed_dim
            )  # h_i, h_j, dot, e_ij, type_emb
        else:
            input_dim = 2 * hidden_dim + 1 + n_rbf

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ShiftedSoftplus(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # RBF for the specific target pairs (re-computed in head)
        self.rbf_fn = GaussianSmearing(start=0.0, stop=Config.CUTOFF, n_rbf=n_rbf)

    def forward(self, x, coords, coupling_pairs, coupling_types):
        """
        Args:
            x: Node embeddings (N_atoms, hidden_dim)
            coords: Atom coordinates (N_atoms, 3)
            coupling_pairs: Indices of interacting pairs (N_couplings, 2)
            coupling_types: Type indices (N_couplings,)
        """
        idx_0, idx_1 = coupling_pairs[:, 0], coupling_pairs[:, 1]

        # Gather node features
        h_0 = x[idx_0]  # (N_couplings, hidden_dim)
        h_1 = x[idx_1]  # (N_couplings, hidden_dim)

        # Compute Dot Product (Multiplicative Interaction)
        dot_prod = (h_0 * h_1).sum(dim=1, keepdim=True)  # (N_couplings, 1)

        # Compute specific edge features for these pairs
        # We re-compute distances here to ensure we have exact features for the target pairs
        # regardless of whether they existed in the radius graph used for message passing.
        diff = coords[idx_0] - coords[idx_1]
        dist = torch.norm(diff, dim=1)
        e_ij = self.rbf_fn(dist)  # (N_couplings, n_rbf)

        # Concatenate features
        features = [h_0, h_1, dot_prod, e_ij]

        if self.use_coupling_embed:
            type_emb = self.type_embedding(coupling_types)  # (N_couplings, embed_dim)
            features.append(type_emb)

        cat_features = torch.cat(features, dim=1)

        # Predict
        out = self.mlp(cat_features)
        return out.squeeze(-1)


class SDIN(nn.Module):
    """
    Scalable Deep Interaction Network.
    """

    def __init__(self):
        super(SDIN, self).__init__()

        # Hyperparameters
        hidden_dim = Config.HIDDEN_DIM
        n_layers = Config.N_LAYERS
        n_rbf = Config.N_RBF
        cutoff = Config.CUTOFF
        num_atom_types = Config.NUM_ATOM_TYPES
        num_coupling_types = Config.NUM_COUPLING_TYPES
        coupling_embed_dim = Config.COUPLING_EMBED_DIM

        self.cutoff = cutoff

        # Atom Embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # RBF Expansion for Graph Edges
        self.rbf_expand = GaussianSmearing(start=0.0, stop=cutoff, n_rbf=n_rbf)

        # Interaction Blocks (Backbone)
        self.interactions = nn.ModuleList(
            [InteractionBlock(hidden_dim, n_rbf) for _ in range(n_layers)]
        )

        # Readout Head
        self.head = ConditionalInteractionHead(
            hidden_dim=hidden_dim,
            n_rbf=n_rbf,
            coupling_embed_dim=coupling_embed_dim,
            n_coupling_types=num_coupling_types,
        )

        # Initialize ShiftedSoftplus appropriately if used
        # (Standard ShiftedSoftplus is log(0.5e^x + 0.5)) - defined here for convenience if not in torch.nn
        if not hasattr(nn, "ShiftedSoftplus"):
            # Monkey patch or define a custom module if strictly needed,
            # but usually standard Softplus is fine or we define a small helper.
            # For this implementation, we will use standard Softplus to avoid import complexity,
            # as ShiftedSoftplus is essentially Softplus - log(2).
            # We will replace the string reference in the classes above with actual Softplus in a real run,
            # or define it here.
            pass

    def forward(
        self,
        atom_types,
        atom_coords,
        batch_index,
        coupling_pairs,
        coupling_types,
        **kwargs
    ):
        """
        Args:
            atom_types: (N_atoms,)
            atom_coords: (N_atoms, 3)
            batch_index: (N_atoms,) assigning atoms to molecules
            coupling_pairs: (N_couplings, 2)
            coupling_types: (N_couplings,)
        """
        # 1. Initial Node Embeddings
        x = self.atom_embedding(atom_types)  # (N_atoms, hidden_dim)

        # 2. Construct Radius Graph (Dynamic Graph Construction)
        # Returns edge_index (2, N_edges)
        edge_index = radius_graph(
            atom_coords, r=self.cutoff, batch=batch_index, max_num_neighbors=100
        )

        # 3. Compute Edge Features
        row, col = edge_index
        diff = atom_coords[row] - atom_coords[col]
        dist = torch.norm(diff, dim=1)
        edge_attr = self.rbf_expand(dist)  # (N_edges, n_rbf)

        # 4. Message Passing Backbone
        for interaction in self.interactions:
            x = interaction(x, edge_index, edge_attr)

        # 5. Readout
        pred = self.head(x, atom_coords, coupling_pairs, coupling_types)

        return pred


# Helper for ShiftedSoftplus (SchNet activation)
# We inject it into nn for the classes above to use it naturally
class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        return F.softplus(x) - self.shift.to(x.device)


setattr(nn, "ShiftedSoftplus", ShiftedSoftplus)
