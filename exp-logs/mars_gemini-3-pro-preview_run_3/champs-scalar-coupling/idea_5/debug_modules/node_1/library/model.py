import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from library.config import Config


class ShiftedSoftplus(nn.Module):
    """
    Shifted Softplus activation function: f(x) = ln(1 + e^x) - ln(2).
    Ensures f(0) is close to 0, which helps signal propagation and convergence
    in deep networks by maintaining activation variance.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(nn.Module):
    """
    Continuous Filter Convolution Layer.
    Generalizes graph convolution by generating filter weights from continuous edge attributes
    (e.g., RBF-expanded distances or angles).
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(CFConv, self).__init__()

        # Filter Generator: Maps geometric features (edge_dim) to interaction weights (hidden_dim)
        # This allows the convolution kernel to be continuous over the geometric domain.
        self.filter_gen = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformation for source nodes before interaction
        self.node_linear = nn.Linear(node_dim, hidden_dim)

        # Transformation for aggregated messages (Update function)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer Normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, node_dim)
            edge_index: Graph connectivity (2, E) -> (source, target)
            edge_attr: Edge features used to generate filters (E, edge_dim)
        """
        src, dst = edge_index

        # 1. Generate dynamic filters from edge attributes
        # W: (E, hidden_dim)
        W = self.filter_gen(edge_attr)

        # 2. Prepare source node features
        # x_j: (E, hidden_dim)
        x_j = self.node_linear(x)[src]

        # 3. Interaction: Element-wise multiplication (Continuous Convolution)
        # message: (E, hidden_dim)
        message = x_j * W

        # 4. Aggregation: Sum messages at target nodes
        # aggr: (N, hidden_dim)
        aggr = scatter(message, dst, dim=0, dim_size=x.size(0), reduce="add")

        # 5. Update: Residual connection + Non-linear transformation
        out = x + self.update_net(aggr)
        out = self.norm(out)

        return out


class InteractionBlock(nn.Module):
    """
    Dual-Graph Interaction Block.
    Propagates information through the Line Graph (Angles) then the Atom Graph (Bonds).
    """

    def __init__(self, hidden_dim, angular_dim):
        super(InteractionBlock, self).__init__()

        # Line Graph Conv: Updates Bond features using Angle features
        # Nodes = Bonds, Edges = Angles
        self.line_conv = CFConv(hidden_dim, angular_dim, hidden_dim)

        # Atom Graph Conv: Updates Atom features using Bond features
        # Nodes = Atoms, Edges = Bonds (where bond features act as the filter input)
        self.atom_conv = CFConv(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, h_atom, h_bond, edge_index, line_edge_index, line_edge_attr):
        # 1. Update Bond Representations (Line Graph Step)
        # h_bond acts as the node features for the line graph
        # line_edge_attr contains the Angular RBF features
        h_bond = self.line_conv(h_bond, line_edge_index, line_edge_attr)

        # 2. Update Atom Representations (Atom Graph Step)
        # h_atom acts as the node features for the atom graph
        # The updated h_bond acts as the "edge attribute" to generate filters for atoms
        h_atom = self.atom_conv(h_atom, edge_index, h_bond)

        return h_atom, h_bond


class DualGraphGNN(nn.Module):
    """
    Stabilized Dual-Graph Network for Scalar Coupling Prediction.
    """

    def __init__(self):
        super(DualGraphGNN, self).__init__()

        # Hyperparameters from Config
        hidden_dim = Config.HIDDEN_DIM
        num_layers = Config.NUM_INTERACTION_LAYERS
        num_rbf_radial = Config.NUM_RBF_RADIAL
        num_rbf_angular = Config.NUM_RBF_ANGULAR
        num_atom_types = len(Config.ATOM_TYPES)

        # --- Embeddings ---
        # Atom Embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # Bond Embedding (Projecting Radial RBF to hidden dim)
        self.bond_embedding = nn.Sequential(
            nn.Linear(num_rbf_radial, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Interaction Blocks ---
        self.blocks = nn.ModuleList(
            [InteractionBlock(hidden_dim, num_rbf_angular) for _ in range(num_layers)]
        )

        # --- Prediction Heads ---
        # 1. Coupling Heads (8 separate MLPs for each type)
        # Input is concatenation of two atom vectors (2 * hidden_dim)
        self.coupling_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(8)
            ]
        )

        # 2. Auxiliary Head: Magnetic Shielding (9 components)
        self.shielding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim // 2, 9),
        )

        # 3. Auxiliary Head: Mulliken Charge (1 component)
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        """
        Forward pass of the model.
        Args:
            data: PyG Data object containing graph attributes.
        Returns:
            pred_coupling: Predicted scalar coupling constants (standardized).
            pred_shielding: Predicted magnetic shielding tensors.
            pred_charge: Predicted mulliken charges.
        """
        # Unpack data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr  # Radial RBF (Distance)
        line_edge_index = data.line_edge_index
        line_edge_attr = data.line_edge_attr  # Angular RBF (Angle)

        # Initialize Embeddings
        h_atom = self.atom_embedding(x)
        h_bond = self.bond_embedding(edge_attr)

        # Message Passing Layers
        for block in self.blocks:
            h_atom, h_bond = block(
                h_atom, h_bond, edge_index, line_edge_index, line_edge_attr
            )

        # --- Coupling Prediction ---
        # Identify atom pairs for coupling
        row = data.edge_index_coupling[0]
        col = data.edge_index_coupling[1]
        types = data.type_coupling

        # Gather features for the pairs
        h_i = h_atom[row]
        h_j = h_atom[col]
        pair_feat = torch.cat([h_i, h_j], dim=-1)

        # Initialize output tensor
        # We use the shape of y_coupling (or id) to ensure correct size
        pred_coupling = torch.zeros(len(types), device=x.device)

        # Apply type-specific heads
        for i in range(8):
            mask = types == i
            if mask.any():
                # Select features for this type
                feat_subset = pair_feat[mask]
                # Run MLP
                out = self.coupling_heads[i](feat_subset)
                # Assign back (squeeze to remove last dim)
                pred_coupling[mask] = out.squeeze(-1)

        # --- Auxiliary Prediction ---
        pred_shielding = self.shielding_head(h_atom)
        pred_charge = self.charge_head(h_atom).squeeze(-1)

        return pred_coupling, pred_shielding, pred_charge
