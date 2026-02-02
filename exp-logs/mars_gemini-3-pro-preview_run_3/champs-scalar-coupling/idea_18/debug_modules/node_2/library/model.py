import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from library.config import Config


class ShiftedSoftplus(nn.Module):
    """
    Activation function used in SchNet: ssp(x) = ln(0.5 * e^x + 0.5).
    Approximated here as softplus(x) - ln(2) so that ssp(0) = 0.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        return F.softplus(x) - self.shift.to(x.device)


class RBFExpansion(nn.Module):
    """
    Expands a distance scalar into a vector of Gaussian Radial Basis Functions.
    """

    def __init__(self, start=0.0, end=5.0, num_centers=50):
        super(RBFExpansion, self).__init__()
        self.start = start
        self.end = end
        self.num_centers = num_centers

        # Compute centers and gamma (width)
        # centers are equally spaced between start and end
        centers = torch.linspace(start, end, num_centers)
        self.register_buffer("centers", centers)

        # Gamma is related to the width of the Gaussian.
        # A common choice is 10 / (end - start), or 1/step_size.
        # Here we use a width such that functions overlap smoothly.
        step = (end - start) / (num_centers - 1)
        self.gamma = 1.0 / step

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (N,) containing distances.
        Returns:
            Tensor of shape (N, num_centers)
        """
        # (N, 1) - (1, num_centers) -> (N, num_centers) via broadcasting
        diff = dist.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class CFConv(nn.Module):
    """
    Continuous Filter Convolution Layer.
    Updates node embeddings based on neighbors and edge distances.
    """

    def __init__(self, hidden_dim, num_rbf):
        super(CFConv, self).__init__()
        self.hidden_dim = hidden_dim

        # Filter generator: Maps RBF edge features to a filter matrix (element-wise)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Linear transformation of neighbor nodes
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output transformation
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr_rbf):
        """
        Args:
            x: Node embeddings (Num_Nodes, Hidden_Dim)
            edge_index: Graph connectivity (2, Num_Edges)
            edge_attr_rbf: RBF expanded edge distances (Num_Edges, Num_RBF)
        """
        src, dst = edge_index

        # 1. Transform neighbor features
        x_in = self.in_proj(x)  # (Num_Nodes, Hidden)

        # 2. Generate continuous filters from edge distances
        W = self.filter_net(edge_attr_rbf)  # (Num_Edges, Hidden)

        # 3. Compute messages: element-wise multiplication
        # Gather source node features for each edge
        x_j = x_in[src]  # (Num_Edges, Hidden)
        messages = x_j * W  # (Num_Edges, Hidden)

        # 4. Aggregate messages to destination nodes
        # scatter_add sums messages where dst index is the same
        aggr_messages = scatter_add(messages, dst, dim=0, dim_size=x.size(0))

        # 5. Update and Residual
        out = self.out_proj(aggr_messages)
        return x + out  # Residual connection


class InteractionAwareHead(nn.Module):
    """
    Readout head that predicts coupling constants.
    Explicitly models pairwise interactions and conditions on coupling type.
    """

    def __init__(self, hidden_dim, num_rbf, coupling_emb_dim, num_coupling_types):
        super(InteractionAwareHead, self).__init__()

        self.use_coupling_emb = Config.USE_COUPLING_EMB

        if self.use_coupling_emb:
            self.type_embedding = nn.Embedding(num_coupling_types, coupling_emb_dim)
            # Input dim: h_i + h_j + (h_i * h_j) + rbf + type_emb
            input_dim = 3 * hidden_dim + num_rbf + coupling_emb_dim
        else:
            input_dim = 3 * hidden_dim + num_rbf

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Separate RBF for the head to ensure we encode the specific coupling distance
        self.rbf = RBFExpansion(
            start=Config.RBF_START, end=Config.RBF_END, num_centers=num_rbf
        )

    def forward(self, x, pos, coupling_atom_index, coupling_type):
        """
        Args:
            x: Node embeddings (Num_Nodes, Hidden_Dim)
            pos: Atom coordinates (Num_Nodes, 3)
            coupling_atom_index: Indices of coupling pairs (2, Num_Couplings)
            coupling_type: Type indices (Num_Couplings,)
        """
        idx_i, idx_j = coupling_atom_index

        # 1. Gather Node Embeddings
        h_i = x[idx_i]
        h_j = x[idx_j]

        # 2. Compute Multiplicative Interaction (Dot Product)
        # Element-wise product captures similarity/interaction
        h_prod = h_i * h_j

        # 3. Compute Exact Distance for Coupling Pair
        # We re-calculate this to ensure we have the precise geometry for the target pair
        # even if it wasn't an edge in the message passing graph.
        pos_i = pos[idx_i]
        pos_j = pos[idx_j]
        dist = (pos_i - pos_j).norm(dim=1)

        # Expand distance
        rbf_feat = self.rbf(dist)

        # 4. Construct Input Vector
        features = [h_i, h_j, h_prod, rbf_feat]

        if self.use_coupling_emb:
            type_emb = self.type_embedding(coupling_type)
            features.append(type_emb)

        cat_features = torch.cat(features, dim=1)

        # 5. Predict
        out = self.mlp(cat_features)
        return out


class MPDIN(nn.Module):
    """
    Molecule-Parallel Deep Interaction Network.
    """

    def __init__(self):
        super(MPDIN, self).__init__()

        # Hyperparameters from Config
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_interactions = Config.NUM_INTERACTIONS
        self.num_rbf = Config.NUM_RBF
        self.coupling_emb_dim = Config.COUPLING_EMB_DIM

        # 1. Atom Embedding
        self.atom_embedding = nn.Embedding(Config.NUM_ATOM_TYPES, self.hidden_dim)

        # 2. RBF Expansion for Backbone
        self.rbf = RBFExpansion(
            start=Config.RBF_START, end=Config.RBF_END, num_centers=self.num_rbf
        )

        # 3. Interaction Blocks (Backbone)
        self.interactions = nn.ModuleList(
            [
                CFConv(self.hidden_dim, self.num_rbf)
                for _ in range(self.num_interactions)
            ]
        )

        # 4. Readout Head
        self.head = InteractionAwareHead(
            hidden_dim=self.hidden_dim,
            num_rbf=self.num_rbf,
            coupling_emb_dim=self.coupling_emb_dim,
            num_coupling_types=Config.NUM_COUPLING_TYPES,
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict: Dictionary containing:
                - x: Atom types (Num_Nodes,)
                - pos: Atom coords (Num_Nodes, 3)
                - edge_index: Radius graph edges (2, Num_Edges)
                - edge_attr: Edge distances (Num_Edges,)
                - coupling_atom_index: Target pair indices (2, Num_Couplings)
                - coupling_type: Target types (Num_Couplings,)
        Returns:
            Tensor of shape (Num_Couplings,) containing predicted scalar coupling constants.
        """
        # Unpack inputs
        atoms = batch_dict["x"]
        pos = batch_dict["pos"]
        edge_index = batch_dict["edge_index"]
        edge_dists = batch_dict["edge_attr"]
        coupling_atom_index = batch_dict["coupling_atom_index"]
        coupling_type = batch_dict["coupling_type"]

        # --- 1. Embedding & Preprocessing ---
        # Embed atom types
        x = self.atom_embedding(atoms)

        # Expand edge distances for the backbone
        edge_rbf = self.rbf(edge_dists)

        # --- 2. Interaction Blocks (Backbone) ---
        for interaction_layer in self.interactions:
            x = interaction_layer(x, edge_index, edge_rbf)

        # --- 3. Readout Head ---
        # Predict for the specific coupling pairs
        # The head handles gathering the updated node embeddings and re-computing
        # the specific distance for the target pair.
        pred = self.head(x, pos, coupling_atom_index, coupling_type)

        return pred.squeeze(-1)
