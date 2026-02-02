import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands a distance scalar into a vector of Gaussian Radial Basis Function values.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Width of the gaussians
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [N] -> [N, 1]
        # offset: [num_gaussians] -> [1, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(nn.Module):
    """
    Continuous Filter Convolution Block.
    Updates node embeddings based on neighbors and edge distances.
    """

    def __init__(self, hidden_dim, num_rbf):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transformation of source nodes before message passing
        self.mlp_atom = nn.Linear(hidden_dim, hidden_dim)

        # Filter generator: Maps edge RBF to a filter matrix (approximated by element-wise mult)
        self.mlp_filter = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformation of aggregated messages
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge RBF features [num_edges, num_rbf]
        """
        row, col = edge_index

        # 1. Generate Filter from Edge Attributes
        W = self.mlp_filter(edge_attr)  # [num_edges, hidden_dim]

        # 2. Transform Source Nodes
        x_j = self.mlp_atom(x)  # [num_nodes, hidden_dim]
        x_j = x_j[col]  # Gather neighbors [num_edges, hidden_dim]

        # 3. Continuous Convolution (Element-wise multiplication)
        m = x_j * W  # [num_edges, hidden_dim]

        # 4. Aggregation (Sum pooling to target nodes)
        # scatter(src, index, dim, dim_size, reduce)
        aggr = scatter(
            m, row, dim=0, dim_size=x.size(0), reduce="add"
        )  # [num_nodes, hidden_dim]

        # 5. Update with Residual Connection
        out = self.mlp_out(aggr)
        return x + out


class SPCFN(nn.Module):
    """
    Scalable Physics-Regularized Continuous Filter Network.
    """

    def __init__(self):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.num_rbf = Config.NUM_RBF

        # -- Embedding Layer --
        self.embedding = nn.Embedding(Config.NUM_ATOM_TYPES, self.hidden_dim)

        # -- Distance Expansion --
        self.rbf = GaussianSmearing(
            start=Config.RBF_START, stop=Config.RBF_END, num_gaussians=self.num_rbf
        )

        # -- Interaction Blocks --
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(self.hidden_dim, self.num_rbf)
                for _ in range(Config.NUM_INTERACTIONS)
            ]
        )

        # -- Auxiliary Heads --
        # Magnetic Shielding (9 components)
        self.aux_shielding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 9),
        )

        # Mulliken Charge (1 component)
        self.aux_charge = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # -- Conditional Readout --
        # Embedding for the coupling type (e.g., 1JHC, 2JHH)
        self.type_embedding = nn.Embedding(Config.NUM_COUPLING_TYPES, self.hidden_dim)

        # Input to readout: Node_i + Node_j + Node_i*Node_j + RBF(dist_ij) + Type_Emb
        # We add the element-wise product (h_i * h_j) to capture pairwise interactions explicitly.
        # Cite solution_lesson_node_00036
        readout_input_dim = (self.hidden_dim * 4) + self.num_rbf

        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, data):
        """
        Args:
            data: Dictionary/Object containing batch data
                - z: Atom types [N]
                - pos: Atom coordinates [N, 3]
                - edge_index: Radius graph indices [2, E]
                - edge_attr: Radius graph distances [E, 1]
                - coupling_index: Indices of coupling pairs [2, C]
                - coupling_type: Coupling type indices [C]
        """
        z = data["z"]
        pos = data["pos"]
        edge_index = data["edge_index"]
        edge_dist = data["edge_attr"]
        coupling_index = data["coupling_index"]
        coupling_type = data["coupling_type"]

        # 1. Initial Embedding
        h = self.embedding(z)  # [N, hidden_dim]

        # 2. Compute Edge RBF for Graph
        # edge_dist is [E, 1], flatten to [E]
        edge_rbf = self.rbf(edge_dist.squeeze(-1))  # [E, num_rbf]

        # 3. Message Passing Layers
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_rbf)

        # 4. Auxiliary Predictions (Regularization)
        pred_shielding = self.aux_shielding(h)  # [N, 9]
        pred_charge = self.aux_charge(h).squeeze(-1)  # [N]

        # 5. Coupling Prediction (Readout)
        # Gather node embeddings for the interacting pairs
        idx_0, idx_1 = coupling_index
        h_0 = h[idx_0]  # [C, hidden_dim]
        h_1 = h[idx_1]  # [C, hidden_dim]

        # Calculate distance for coupling pairs specifically
        # (Coupling pairs might be further than cutoff, so we compute on fly)
        pos_0 = pos[idx_0]
        pos_1 = pos[idx_1]
        dist_c = torch.norm(pos_0 - pos_1, dim=-1)  # [C]
        rbf_c = self.rbf(dist_c)  # [C, num_rbf]

        # Get coupling type embedding
        type_emb = self.type_embedding(coupling_type)  # [C, hidden_dim]

        # Concatenate all features
        # [h_i, h_j, h_i * h_j, rbf_ij, type_emb]
        # Explicit interaction term improves convergence for pairwise properties.
        out_input = torch.cat([h_0, h_1, h_0 * h_1, rbf_c, type_emb], dim=-1)

        # Final prediction
        pred_coupling = self.readout(out_input).squeeze(-1)  # [C]

        return pred_coupling, pred_shielding, pred_charge
