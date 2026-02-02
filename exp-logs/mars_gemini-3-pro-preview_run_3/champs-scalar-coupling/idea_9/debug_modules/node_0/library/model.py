import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from library.config import Config


class RBFExpansion(nn.Module):
    """
    Expands a scalar feature into a vector of Gaussian Radial Basis Functions.
    Used for encoding distances and angles.
    """

    def __init__(self, min_value, max_value, num_rbf, trainable=False):
        super().__init__()
        self.num_rbf = num_rbf

        # Initialize centers linearly spaced between min and max
        centers = torch.linspace(min_value, max_value, num_rbf)

        # Calculate width (gamma) based on spacing
        if num_rbf > 1:
            gap = centers[1] - centers[0]
            gamma = 1.0 / (gap**2)
        else:
            gamma = 1.0

        if trainable:
            self.centers = nn.Parameter(centers)
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Scalar features (N,).
        Returns:
            torch.Tensor: Expanded features (N, num_rbf).
        """
        x = x.view(-1, 1)
        # exp(-gamma * (x - mu)^2)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)


class InteractionBlock(nn.Module):
    """
    A single layer of Directional Message Passing.
    Updates edge embeddings based on angular interactions with preceding edges,
    and updates node embeddings based on incoming edges.
    """

    def __init__(self, hidden_dim, num_angle_rbf):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Edge Message Function (incorporating angles)
        # Input: e_kj (hidden) + angle_rbf (num_angle_rbf)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_angle_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Edge Update Function
        # Input: e_ji (hidden) + aggregated_messages (hidden)
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Node Update Function
        # Input: h_i (hidden) + aggregated_edges (hidden)
        self.node_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer Norms for stability
        self.layer_norm_edge = nn.LayerNorm(hidden_dim)
        self.layer_norm_node = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_attr, edge_index, triplet_indices, triplet_angles_rbf):
        """
        Args:
            h (torch.Tensor): Node embeddings (NumNodes, Hidden).
            edge_attr (torch.Tensor): Edge embeddings e_ji (NumEdges, Hidden).
            edge_index (torch.Tensor): Edge connectivity (2, NumEdges). Row 0: src, Row 1: dst.
            triplet_indices (torch.Tensor): Triplet indices (2, NumTriplets).
                                            Row 0: index of edge k->j.
                                            Row 1: index of edge j->i.
            triplet_angles_rbf (torch.Tensor): Expanded angle features (NumTriplets, AngleRBF).

        Returns:
            h_new, edge_attr_new
        """
        # --- 1. Angular Message Passing (Edge Update) ---
        if triplet_indices.numel() > 0:
            # Gather features of incoming edges k->j
            # triplet_indices[0] contains indices of k->j
            incoming_edges = edge_attr[triplet_indices[0]]  # (Triplets, Hidden)

            # Combine with angular features
            msg_input = torch.cat([incoming_edges, triplet_angles_rbf], dim=-1)

            # Compute messages
            messages = self.message_mlp(msg_input)  # (Triplets, Hidden)

            # Aggregate messages to the target edge j->i
            # triplet_indices[1] contains indices of j->i
            # We sum all messages destined for the same edge
            aggr_messages = scatter_sum(
                messages, triplet_indices[1], dim=0, dim_size=edge_attr.size(0)
            )
        else:
            aggr_messages = torch.zeros_like(edge_attr)

        # Update edge embeddings with residual connection
        edge_input = torch.cat([edge_attr, aggr_messages], dim=-1)
        edge_update = self.edge_update_mlp(edge_input)
        edge_attr_new = self.layer_norm_edge(edge_attr + edge_update)

        # --- 2. Node Update ---
        # Aggregate incoming edges to nodes
        # edge_index[1] contains the target node indices (i for edge j->i)
        aggr_edges = scatter_sum(
            edge_attr_new, edge_index[1], dim=0, dim_size=h.size(0)
        )

        # Update node embeddings with residual connection
        node_input = torch.cat([h, aggr_edges], dim=-1)
        node_update = self.node_update_mlp(node_input)
        h_new = self.layer_norm_node(h + node_update)

        return h_new, edge_attr_new


class TypeSpecificHeads(nn.Module):
    """
    A collection of MLP heads, one for each scalar coupling type.
    Routes input features to the appropriate head based on the type index.
    """

    def __init__(self, input_dim, num_types):
        super().__init__()
        self.num_types = num_types

        # Create a ModuleList of heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )
                for _ in range(num_types)
            ]
        )

    def forward(self, features, types):
        """
        Args:
            features (torch.Tensor): Input features (N, InputDim).
            types (torch.Tensor): Type indices (N,).

        Returns:
            torch.Tensor: Predictions (N, 1).
        """
        output = torch.zeros(features.size(0), 1, device=features.device)

        # Iterate over unique types present in the batch for efficiency
        unique_types = torch.unique(types)
        for t in unique_types:
            t_idx = t.item()
            mask = types == t
            if mask.any():
                # Apply the specific head to the masked features
                output[mask] = self.heads[t_idx](features[mask])

        return output


class MoleculeModel(nn.Module):
    """
    Main Directional Message Passing Neural Network.

    Architecture:
    1. Initial Embedding (Nodes & Edges)
    2. Stack of InteractionBlocks (Message Passing)
    3. Readout Heads (Coupling Type Specific + Aux Tasks)
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_interactions = Config.NUM_INTERACTIONS
        self.num_rbf = Config.NUM_RBF
        self.num_angle_rbf = Config.NUM_ANGLE_RBF
        self.cutoff = Config.CUTOFF

        # --- 1. Feature Expansion & Embedding ---
        # Distance RBF: 0 to Cutoff
        self.dist_rbf = RBFExpansion(0.0, self.cutoff, self.num_rbf)

        # Angle RBF: -1 to 1 (Cosine of angle)
        self.angle_rbf = RBFExpansion(-1.0, 1.0, self.num_angle_rbf)

        # Atom Embedding
        self.atom_embedding = nn.Embedding(Config.NUM_ATOM_TYPES, self.hidden_dim)

        # Edge Embedding Initialization
        # Concatenates: RBF(dist) + NodeEmb(src) + NodeEmb(dst)
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.num_rbf + 2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # --- 2. Message Passing ---
        self.layers = nn.ModuleList(
            [
                InteractionBlock(self.hidden_dim, self.num_angle_rbf)
                for _ in range(self.num_interactions)
            ]
        )

        # --- 3. Readout Heads ---
        # Primary Task: Scalar Coupling Constant
        # Input: h_u || h_v || e_uv
        readout_dim = 3 * self.hidden_dim
        self.coupling_heads = TypeSpecificHeads(readout_dim, len(Config.COUPLING_TYPES))

        # Auxiliary Task: Mulliken Charges (Node Level)
        self.charge_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )

        # Auxiliary Task: Magnetic Shielding (Node Level)
        self.shielding_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 9)
        )

        # Auxiliary Task: Dipole Moment (Graph Level)
        self.dipole_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )

        # Auxiliary Task: Potential Energy (Graph Level)
        self.potential_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (dict): Batch dictionary from collate_batch.

        Returns:
            dict: Dictionary containing predictions for primary and aux tasks.
        """
        # Unpack necessary tensors
        node_z = batch["node_z"]
        edge_index = batch["edge_index"]
        edge_dist = batch["edge_dist"]
        triplet_indices = batch["triplet_indices"]
        triplet_angles = batch["triplet_angles"]
        target_indices = batch["target_indices"]
        target_types = batch["target_types"]

        # --- 1. Initialization ---
        # Embed Nodes
        h = self.atom_embedding(node_z)  # (N, H)

        # Embed Edges
        dist_feat = self.dist_rbf(edge_dist)  # (E, RBF)

        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        # e_ji initial = MLP(RBF(d_ji) || h_j || h_i)
        edge_input = torch.cat([dist_feat, h_src, h_dst], dim=-1)
        e = self.edge_embedding(edge_input)  # (E, H)

        # Prepare Angle Features
        if triplet_indices.numel() > 0:
            # We use cosine of the angle for RBF expansion
            # triplet_angles are in radians
            angle_cos = torch.cos(triplet_angles)
            angle_feat = self.angle_rbf(angle_cos)  # (T, AngleRBF)
        else:
            angle_feat = torch.empty(0, self.num_angle_rbf, device=h.device)

        # --- 2. Message Passing ---
        for layer in self.layers:
            h, e = layer(h, e, edge_index, triplet_indices, angle_feat)

        # --- 3. Predictions ---

        # A. Scalar Coupling (Primary Task)
        # target_indices points to the edge index corresponding to the pair
        # We retrieve the learned edge embedding e_uv
        e_target = e[target_indices]

        # We also retrieve the node embeddings for u and v
        # edge_index[0] is src, edge_index[1] is dst
        u_idx = edge_index[0, target_indices]
        v_idx = edge_index[1, target_indices]

        h_u = h[u_idx]
        h_v = h[v_idx]

        # Concatenate: h_u || h_v || e_uv
        coupling_feat = torch.cat([h_u, h_v, e_target], dim=-1)

        # Pass to type-specific heads
        coupling_pred = self.coupling_heads(coupling_feat, target_types)

        # B. Auxiliary Tasks
        # Node-level
        charges_pred = self.charge_head(h)
        shielding_pred = self.shielding_head(h)

        # Graph-level (Sum Pooling)
        node_batch = batch["node_batch"]
        batch_size = batch["batch_size"]

        h_graph = scatter_sum(h, node_batch, dim=0, dim_size=batch_size)

        dipole_pred = self.dipole_head(h_graph)
        potential_pred = self.potential_head(h_graph)

        return {
            "scalar_coupling": coupling_pred,
            "charges": charges_pred,
            "shielding": shielding_pred,
            "dipole": dipole_pred,
            "potential": potential_pred,
        }
