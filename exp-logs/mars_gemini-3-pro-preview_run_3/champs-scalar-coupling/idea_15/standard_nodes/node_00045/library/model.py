import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from library.config import (
    HIDDEN_DIM,
    NUM_INTERACTIONS,
    RBF_RADIUS,
    NUM_RBF,
    READOUT_HIDDEN_DIM,
    NUM_ATOM_TYPES,
    NUM_COUPLING_TYPES,
)
from library.utils import RBFExpansion


class ContinuousFilterConv(nn.Module):
    """
    Continuous Filter Convolution Layer.
    Generates filter weights from edge features (RBF distances) and applies them
    to node features in a continuous manner.
    """

    def __init__(self, hidden_dim, num_rbf):
        super(ContinuousFilterConv, self).__init__()
        self.hidden_dim = hidden_dim

        # Filter Generator Network: RBF -> Filter
        # Maps geometric information (distance) to interaction weights
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input transformation for source nodes
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Expanded RBF edge features [E, num_rbf]
        """
        # 1. Generate Filters from Edge Attributes
        # W: [E, hidden_dim]
        W = self.filter_network(edge_attr)

        # 2. Transform Source Node Features
        # x_in: [N, hidden_dim]
        x_in = self.in_proj(x)

        # 3. Message Passing
        # Gather features of neighbors: x_j = x[source_indices]
        # edge_index[0] is source, edge_index[1] is target
        x_j = x_in[edge_index[0]]  # [E, hidden_dim]

        # Apply continuous filter (Element-wise multiplication)
        msg = x_j * W  # [E, hidden_dim]

        # Aggregate messages at target nodes (Sum)
        out = scatter_add(msg, edge_index[1], dim=0, dim_size=x.size(0))

        return out


class InteractionBlock(nn.Module):
    """
    Standard Interaction Block (SchNet-style).
    Combines Continuous Filter Convolution with atom-wise dense layers and residual connections.
    """

    def __init__(self, hidden_dim, num_rbf):
        super(InteractionBlock, self).__init__()
        self.conv = ContinuousFilterConv(hidden_dim, num_rbf)

        # Atom-wise processing (MLP)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # Residual connection path
        identity = x

        # Convolution
        v = self.conv(x, edge_index, edge_attr)

        # Atom-wise non-linear update
        v = self.dense(v)

        # Add residual
        return identity + v


class InteractionAwareReadout(nn.Module):
    """
    Interaction-Aware Shared Conditional Head.
    Predicts coupling constants by combining node embeddings, explicit interaction terms,
    coupling type embeddings, and explicit edge distances.
    """

    def __init__(self, hidden_dim, num_coupling_types, num_rbf, output_dim=1):
        super(InteractionAwareReadout, self).__init__()

        # Learnable embedding for the coupling type (e.g., 1JHC, 2JHH)
        self.type_embedding = nn.Embedding(num_coupling_types, hidden_dim)

        # Input dimension construction:
        # Node i (hidden_dim) + Node j (hidden_dim) + Dot Product (1) + Type Emb (hidden_dim) + Edge RBF (num_rbf)
        input_dim = 3 * hidden_dim + 1 + num_rbf

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, READOUT_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(READOUT_HIDDEN_DIM, READOUT_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(READOUT_HIDDEN_DIM, output_dim),
        )

    def forward(self, h, coupling_atom_index, coupling_type, coupling_rbf):
        """
        Args:
            h: Final node embeddings [N, hidden_dim]
            coupling_atom_index: Indices of atom pairs [2, K]
            coupling_type: Coupling type indices [K]
            coupling_rbf: RBF expansion of distance between pairs [K, num_rbf]
        """
        # Gather node embeddings for the interacting pairs
        idx0 = coupling_atom_index[0]
        idx1 = coupling_atom_index[1]

        h0 = h[idx0]  # [K, hidden_dim]
        h1 = h[idx1]  # [K, hidden_dim]

        # Explicit Multiplicative Interaction (Dot Product)
        dot = (h0 * h1).sum(dim=1, keepdim=True)  # [K, 1]

        # Get coupling type embedding
        t_emb = self.type_embedding(coupling_type)  # [K, hidden_dim]

        # Concatenate all features including explicit edge distance (Cite solution_lesson_node_00011)
        cat_features = torch.cat([h0, h1, dot, t_emb, coupling_rbf], dim=1)

        # Predict
        out = self.mlp(cat_features)
        return out.squeeze(-1)


class MPIN(nn.Module):
    """
    Molecule-Parallel Interaction Network (MP-IN).
    A Node-Centric GNN backbone with an Interaction-Aware Readout.
    """

    def __init__(self):
        super(MPIN, self).__init__()

        # 1. Atom Type Embedding
        self.embedding = nn.Embedding(NUM_ATOM_TYPES, HIDDEN_DIM)

        # 2. RBF Expansion for Edge Distances
        # Used to create continuous filters for the backbone
        self.rbf_expansion = RBFExpansion(vmin=0.0, vmax=RBF_RADIUS, bins=NUM_RBF)

        # 3. Backbone: Stack of Interaction Blocks
        self.interactions = nn.ModuleList(
            [InteractionBlock(HIDDEN_DIM, NUM_RBF) for _ in range(NUM_INTERACTIONS)]
        )

        # 4. Readout Head
        self.readout = InteractionAwareReadout(HIDDEN_DIM, NUM_COUPLING_TYPES, NUM_RBF)

        # 5. Auxiliary Heads (Cite solution_lesson_node_00024)
        # Mulliken Charges (scalar per atom)
        self.charge_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        # Magnetic Shielding (9 components per atom)
        self.shielding_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 9),
        )

    def forward(self, batch):
        """
        Forward pass for a batch of molecules.
        Args:
            batch: Dictionary containing collated graph data.
        """
        # Unpack batch data
        node_types = batch["node_types"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]  # Raw distances [E, 1]
        coupling_atom_index = batch["coupling_atom_index"]
        coupling_type = batch["coupling_type"]
        coupling_dist = batch["coupling_dist"]  # [K, 1]

        # 1. Initial Node Embedding
        h = self.embedding(node_types)  # [N, hidden_dim]

        # 2. Expand Edge Distances
        # Convert scalar distances to RBF vectors
        edge_rbf = self.rbf_expansion(edge_attr.squeeze(-1))  # [E, num_rbf]

        # 3. Backbone Message Passing
        # Update node embeddings based on geometric graph structure
        for block in self.interactions:
            h = block(h, edge_index, edge_rbf)

        # 4. Main Readout
        # Expand coupling distances for the readout
        coupling_rbf = self.rbf_expansion(coupling_dist.squeeze(-1))
        pred_coupling = self.readout(
            h, coupling_atom_index, coupling_type, coupling_rbf
        )

        # 5. Auxiliary Readouts
        pred_charges = self.charge_head(h).squeeze(-1)
        pred_shielding = self.shielding_head(h)

        return {
            "coupling": pred_coupling,
            "charges": pred_charges,
            "shielding": pred_shielding,
        }
