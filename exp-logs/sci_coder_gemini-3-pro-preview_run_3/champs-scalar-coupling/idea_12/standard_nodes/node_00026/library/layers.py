import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from library.config import Config


class RBFExpansion(nn.Module):
    """
    Expands a scalar feature into a vector of Radial Basis Functions.
    Used for encoding distances and angles.
    """

    def __init__(self, start, stop, num_gaussians):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians

        # Create centers for the Gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        # Register as buffer so it's part of state_dict but not a parameter
        self.register_buffer("offset", offset)

        # Calculate width (gamma) based on spacing
        # width = step_size. gamma = 0.5 / width^2
        step = (stop - start) / num_gaussians
        self.coeff = -0.5 / (step**2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,) containing scalar values.
        Returns:
            Tensor of shape (N, num_gaussians)
        """
        # Expand dims for broadcasting: (N, 1) - (1, num_gaussians)
        diff = x.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(diff, 2))


class InteractionBlock(nn.Module):
    """
    Directional Message Passing Layer with Continuous Filter Convolutions.
    Updates edge embeddings based on neighboring edges and geometric features (distance, angle).
    """

    def __init__(self, hidden_dim, num_rbf, num_angle_rbf, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Filter Generator
        # Generates weights based on geometry (distance of source edge + angle between edges)
        # Input: [RBF(dist_kj), RBF(angle_kji)]
        self.filter_mlp = nn.Sequential(
            nn.Linear(num_rbf + num_angle_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Update Network
        # Combines current edge embedding with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # 3. Normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, triplet_indices, rbf_dist_src, rbf_angle):
        """
        Args:
            h: Current edge embeddings (Num_Edges, Hidden_Dim)
            triplet_indices: (2, Num_Triplets).
                             Row 0: Indices of source edges (k->j)
                             Row 1: Indices of target edges (j->i)
            rbf_dist_src: RBF expansion of source edge lengths (Num_Triplets, Num_RBF)
            rbf_angle: RBF expansion of bond angles (Num_Triplets, Num_Angle_RBF)
        """
        # --- 1. Generate Dynamic Filters ---
        # Concatenate geometric features of the interaction triplet
        geom_features = torch.cat([rbf_dist_src, rbf_angle], dim=-1)
        filters = self.filter_mlp(geom_features)  # (Num_Triplets, Hidden_Dim)

        # --- 2. Gather Source Messages ---
        # Get embeddings of the "incoming" edges in the triplets
        src_edge_indices = triplet_indices[0]
        src_h = h[src_edge_indices]  # (Num_Triplets, Hidden_Dim)

        # Apply filter (Hadamard product)
        messages = src_h * filters

        # --- 3. Aggregate Messages ---
        # Sum messages destined for the same target edge
        target_edge_indices = triplet_indices[1]
        aggregated = scatter(
            messages, target_edge_indices, dim=0, dim_size=h.size(0), reduce="sum"
        )

        # --- 4. Update State ---
        # Concatenate original state with aggregated messages
        out = torch.cat([h, aggregated], dim=-1)
        out = self.update_mlp(out)

        # Residual connection + LayerNorm
        return self.norm(h + out)


class ScalarCouplingModel(nn.Module):
    """
    Main model class implementing Scalable Directional Message Passing.
    Includes shared conditional readout and auxiliary heads.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.atom_embedding = nn.Embedding(config.NUM_ATOM_TYPES, config.HIDDEN_DIM)
        self.type_embedding = nn.Embedding(config.NUM_COUPLING_TYPES, config.HIDDEN_DIM)

        # --- Geometric Expansion ---
        # Distance RBF: 0 to Cutoff
        self.dist_rbf = RBFExpansion(0.0, config.CUTOFF, config.NUM_RBF)
        # Angle RBF: -1 to 1 (Cosine)
        self.angle_rbf = RBFExpansion(-1.0, 1.0, config.NUM_ANGLE_RBF)

        # --- Initialization Layer ---
        # Initializes edge embeddings from node features and bond length
        self.edge_init = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2 + config.NUM_RBF, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
        )

        # --- Message Passing Layers ---
        self.layers = nn.ModuleList(
            [
                InteractionBlock(
                    config.HIDDEN_DIM,
                    config.NUM_RBF,
                    config.NUM_ANGLE_RBF,
                    config.DROPOUT,
                )
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # --- Readout Heads ---

        # 1. Primary Task: Scalar Coupling Prediction
        # Input: [Node_i, Node_j, Edge_ij, Type_Embedding]
        readout_input_dim = config.HIDDEN_DIM * 3 + config.HIDDEN_DIM
        self.coupling_head = nn.Sequential(
            nn.Linear(readout_input_dim, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
        )

        # 2. Auxiliary Tasks (Physics Regularization)
        if config.USE_AUXILIARY_HEADS:
            # Magnetic Shielding (9 components per atom)
            self.shielding_head = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.SiLU(),
                nn.Linear(config.HIDDEN_DIM // 2, 9),
            )
            # Mulliken Charges (1 component per atom)
            self.charge_head = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.SiLU(),
                nn.Linear(config.HIDDEN_DIM // 2, 1),
            )

    def forward(
        self,
        atom_types,
        edge_index,
        edge_dist,
        triplet_index,
        triplet_angle,
        coupling_node_indices,
        coupling_edge_indices,
        coupling_types,
    ):
        """
        Forward pass of the model.

        Args:
            atom_types: (N_atoms,) LongTensor
            edge_index: (2, N_edges) LongTensor (Source, Target)
            edge_dist: (N_edges,) FloatTensor (Distances)
            triplet_index: (2, N_triplets) LongTensor (Edge_k->j, Edge_j->i)
            triplet_angle: (N_triplets,) FloatTensor (Cosine of angles)
            coupling_node_indices: (N_couplings, 2) LongTensor
            coupling_edge_indices: (N_couplings,) LongTensor (Index of edge connecting pair)
            coupling_types: (N_couplings,) LongTensor (Type indices)

        Returns:
            Dictionary containing 'coupling', 'shielding', and 'charge' predictions.
        """

        # --- 1. Initial Embeddings ---
        # Node Features
        node_h = self.atom_embedding(atom_types)  # (N_atoms, H)

        # Edge Features Initialization
        # e_ij = MLP(h_i, h_j, RBF(d_ij))
        row, col = edge_index
        rbf_d = self.dist_rbf(edge_dist)  # (N_edges, Num_RBF)

        edge_input = torch.cat([node_h[row], node_h[col], rbf_d], dim=-1)
        edge_h = self.edge_init(edge_input)  # (N_edges, H)

        # Precompute Angle RBFs
        rbf_a = self.angle_rbf(triplet_angle)  # (N_triplets, Num_Angle_RBF)

        # --- 2. Directional Message Passing ---
        for layer in self.layers:
            # We need the distance RBF of the *source* edge in every triplet
            # triplet_index[0] contains indices of edges k->j
            rbf_d_src = rbf_d[triplet_index[0]]

            edge_h = layer(edge_h, triplet_index, rbf_d_src, rbf_a)

        # --- 3. Node Aggregation ---
        # Update node embeddings by aggregating incoming edge embeddings
        # h_i_new = h_i + sum_{j} e_{ji}
        # edge_index[1] is the target node index
        edge_agg = scatter(
            edge_h, edge_index[1], dim=0, dim_size=node_h.size(0), reduce="add"
        )
        node_h_out = node_h + edge_agg

        # --- 4. Readout ---

        # Primary Task: Coupling Constant
        # Gather features for the specific pairs we want to predict
        idx_i = coupling_node_indices[:, 0]
        idx_j = coupling_node_indices[:, 1]

        h_i = node_h_out[idx_i]
        h_j = node_h_out[idx_j]

        # Gather the specific edge embedding connecting the pair
        # Note: We assume valid edges exist (radius graph cutoff >= coupling dist)
        e_ij = edge_h[coupling_edge_indices]

        # Coupling Type Embedding
        t_emb = self.type_embedding(coupling_types)

        # Concatenate and Predict
        # Injecting edge embedding e_ij is crucial for geometric precision
        coupling_input = torch.cat([h_i, h_j, e_ij, t_emb], dim=-1)
        pred_coupling = self.coupling_head(coupling_input)

        # Auxiliary Tasks
        pred_shielding = None
        pred_charge = None

        if self.config.USE_AUXILIARY_HEADS:
            pred_shielding = self.shielding_head(node_h_out)
            pred_charge = self.charge_head(node_h_out)

        return {
            "coupling": pred_coupling,
            "shielding": pred_shielding,
            "charge": pred_charge,
        }


def get_model(config=None):
    """Helper to instantiate the model with default or provided config."""
    if config is None:
        config = Config()
    return ScalarCouplingModel(config)
