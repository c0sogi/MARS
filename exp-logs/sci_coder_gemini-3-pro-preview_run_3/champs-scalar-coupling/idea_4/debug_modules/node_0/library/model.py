import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from library.config import Config
from library.features import GaussianSmearing


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron with LayerNorm and SILU activation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LineGNNLayer(nn.Module):
    """
    Message Passing Layer for the Line Graph.
    Updates Bond (Edge) embeddings based on angular interactions with neighboring bonds.
    """

    def __init__(self, hidden_dim, angle_rbf_dim, dropout=0.0):
        super(LineGNNLayer, self).__init__()

        # Message function: Combines source bond, dest bond, and angle info
        # Input: Bond_i (H) + Bond_j (H) + Angle (RBF)
        self.message_mlp = MLP(
            input_dim=2 * hidden_dim + angle_rbf_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Update function: Combines current bond with aggregated messages
        self.update_mlp = MLP(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, edge_emb, line_edge_index, line_edge_attr):
        """
        Args:
            edge_emb: (E, H) - Current embeddings of Atom Graph edges (nodes in Line Graph).
            line_edge_index: (2, L) - Connections between bonds (triplets).
            line_edge_attr: (L, A) - Expanded angle features.
        """
        src, dst = line_edge_index

        # Gather source and destination bond embeddings
        # src_emb: (L, H), dst_emb: (L, H)
        src_emb = edge_emb[src]
        dst_emb = edge_emb[dst]

        # 1. Compute Messages
        # Cat: [Bond_u, Bond_v, Angle_uv]
        raw_msg = torch.cat([src_emb, dst_emb, line_edge_attr], dim=1)
        messages = self.message_mlp(raw_msg)  # (L, H)

        # 2. Aggregate Messages
        # Sum messages flowing into each destination bond
        # dim_size=edge_emb.size(0) ensures we cover all edges even if some have no angular neighbors
        aggr_msg = scatter_add(
            messages, dst, dim=0, dim_size=edge_emb.size(0)
        )  # (E, H)

        # 3. Update Bond Embeddings (Residual)
        update_input = torch.cat([edge_emb, aggr_msg], dim=1)
        out_emb = edge_emb + self.update_mlp(update_input)

        return out_emb


class AtomGNNLayer(nn.Module):
    """
    Message Passing Layer for the Atom Graph.
    Updates Atom (Node) embeddings based on interactions with neighbors and bonds.
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super(AtomGNNLayer, self).__init__()

        # Message function: Combines source atom, dest atom, and connecting bond
        # Input: Atom_i (H) + Atom_j (H) + Bond_ij (H)
        self.message_mlp = MLP(
            input_dim=3 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Update function
        self.update_mlp = MLP(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_emb):
        """
        Args:
            x: (N, H) - Current atom embeddings.
            edge_index: (2, E) - Connectivity.
            edge_emb: (E, H) - Current bond embeddings.
        """
        src, dst = edge_index

        # Gather atom embeddings
        src_x = x[src]
        dst_x = x[dst]

        # 1. Compute Messages
        # Cat: [Atom_u, Atom_v, Bond_uv]
        raw_msg = torch.cat([src_x, dst_x, edge_emb], dim=1)
        messages = self.message_mlp(raw_msg)  # (E, H)

        # 2. Aggregate Messages
        # Sum messages flowing into each destination atom
        aggr_msg = scatter_add(messages, dst, dim=0, dim_size=x.size(0))  # (N, H)

        # 3. Update Atom Embeddings (Residual)
        update_input = torch.cat([x, aggr_msg], dim=1)
        out_x = x + self.update_mlp(update_input)

        return out_x


class InteractionBlock(nn.Module):
    """
    Combines Line Graph convolution and Atom Graph convolution.
    Flow: Angles -> Bonds -> Atoms.
    """

    def __init__(self, hidden_dim, angle_rbf_dim, dropout=0.0):
        super(InteractionBlock, self).__init__()
        self.line_conv = LineGNNLayer(hidden_dim, angle_rbf_dim, dropout)
        self.atom_conv = AtomGNNLayer(hidden_dim, dropout)

    def forward(self, x, edge_index, edge_emb, line_edge_index, line_edge_attr):
        # 1. Update Bonds using Angular info
        edge_emb = self.line_conv(edge_emb, line_edge_index, line_edge_attr)

        # 2. Update Atoms using Bond info
        x = self.atom_conv(x, edge_index, edge_emb)

        return x, edge_emb


class DualGraphNetwork(nn.Module):
    """
    End-to-End Dual Graph Network for Scalar Coupling Prediction.
    Includes type-specific heads and auxiliary physics-aware tasks.
    """

    def __init__(self):
        super(DualGraphNetwork, self).__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout = Config.DROPOUT

        # ---------------------------------------------------------
        # 1. Embeddings
        # ---------------------------------------------------------
        # Atom Embedding (H, C, N, O, F -> 0..4)
        self.atom_embedding = nn.Embedding(5, self.hidden_dim)

        # Distance Expansion (RBF)
        self.distance_rbf = GaussianSmearing(
            start=Config.RBF_DISTANCE_MIN,
            stop=Config.RBF_DISTANCE_MAX,
            num_gaussians=Config.NUM_RBF_DISTANCE,
        )
        self.edge_embedding = nn.Linear(Config.NUM_RBF_DISTANCE, self.hidden_dim)

        # Angle Expansion (RBF) - Input is cos(theta) in [-1, 1]
        self.angle_rbf = GaussianSmearing(
            start=-1.0, stop=1.0, num_gaussians=Config.NUM_RBF_ANGLE
        )
        self.angle_embedding = nn.Linear(
            Config.NUM_RBF_ANGLE, self.hidden_dim
        )  # Project RBF to hidden

        # ---------------------------------------------------------
        # 2. Backbone (Interaction Blocks)
        # ---------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=self.hidden_dim,
                    angle_rbf_dim=self.hidden_dim,  # We project RBF to hidden before passing
                    dropout=self.dropout,
                )
                for _ in range(Config.NUM_INTERACTION_LAYERS)
            ]
        )

        # ---------------------------------------------------------
        # 3. Heads
        # ---------------------------------------------------------

        # Coupling Heads (One per type)
        # Input: Cat(Atom_i, Atom_j) -> 2 * hidden_dim
        self.coupling_heads = nn.ModuleList(
            [
                MLP(
                    2 * self.hidden_dim,
                    self.hidden_dim,
                    1,
                    num_layers=3,
                    dropout=self.dropout,
                )
                for _ in range(Config.NUM_COUPLING_TYPES)
            ]
        )

        # Auxiliary Head: Magnetic Shielding (9 components)
        self.shielding_head = MLP(
            self.hidden_dim, self.hidden_dim // 2, 9, num_layers=2
        )

        # Auxiliary Head: Mulliken Charges (1 component)
        self.charge_head = MLP(self.hidden_dim, self.hidden_dim // 2, 1, num_layers=2)

    def forward(self, data):
        """
        Args:
            data: DualGraphData object containing batch data.
        Returns:
            pred_coupling: (N_targets, 1)
            pred_shielding: (N_nodes, 9)
            pred_charges: (N_nodes, 1)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        line_edge_index = data.line_edge_index
        line_edge_attr = data.line_edge_attr

        # ---------------------------------------------------------
        # 1. Initial Embeddings
        # ---------------------------------------------------------
        # Nodes
        h_nodes = self.atom_embedding(x)  # (N, H)

        # Edges (Distances)
        edge_rbf = self.distance_rbf(edge_attr)  # (E, RBF_dist)
        h_edges = self.edge_embedding(edge_rbf)  # (E, H)

        # Line Edges (Angles)
        angle_rbf = self.angle_rbf(line_edge_attr)  # (L, RBF_angle)
        h_angles = self.angle_embedding(angle_rbf)  # (L, H)

        # ---------------------------------------------------------
        # 2. Message Passing
        # ---------------------------------------------------------
        for layer in self.layers:
            h_nodes, h_edges = layer(
                h_nodes, edge_index, h_edges, line_edge_index, h_angles
            )

        # ---------------------------------------------------------
        # 3. Auxiliary Predictions
        # ---------------------------------------------------------
        pred_shielding = self.shielding_head(h_nodes)
        pred_charges = self.charge_head(h_nodes)

        # ---------------------------------------------------------
        # 4. Coupling Prediction
        # ---------------------------------------------------------
        # We need to predict couplings for specific pairs given in target_index
        # target_index: (2, N_targets)
        # target_type: (N_targets,)

        target_idx = data.target_index
        target_type = data.target_type

        # Gather node features for the pairs
        # idx_0, idx_1 = target_idx[0], target_idx[1]
        h_0 = h_nodes[target_idx[0]]
        h_1 = h_nodes[target_idx[1]]

        # Construct pair feature
        pair_emb = torch.cat([h_0, h_1], dim=1)  # (N_targets, 2*H)

        # Initialize output
        pred_coupling = torch.zeros(pair_emb.size(0), 1, device=pair_emb.device)

        # Route to specific heads based on type
        for type_idx in range(Config.NUM_COUPLING_TYPES):
            mask = target_type == type_idx
            if mask.any():
                # Select subset
                subset_emb = pair_emb[mask]
                # Apply head
                subset_pred = self.coupling_heads[type_idx](subset_emb)
                # Scatter back
                pred_coupling[mask] = subset_pred

        return pred_coupling, pred_shielding, pred_charges
