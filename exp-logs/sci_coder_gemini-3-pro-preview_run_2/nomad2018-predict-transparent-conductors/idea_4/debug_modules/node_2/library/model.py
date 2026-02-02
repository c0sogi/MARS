import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Batch
from library.config import Config
from library.utils import GaussianRBF, Standardizer, compute_rmsle


class GatedGCNLayer(MessagePassing):
    """
    Gated Graph Convolution Layer.
    Updates node features based on neighbor features and edge gating.
    This layer is used for both the Atom Graph (updating atoms via bonds)
    and the Line Graph (updating bonds via angles).
    """

    def __init__(self, node_dim, edge_dim, dropout=0.0):
        super().__init__(aggr="add")  # Aggregation method: Sum

        # Gating mechanism: Computes a soft gate (0-1) for each message
        # gate_ij = sigmoid(W_src * x_i + W_dst * x_j + W_edge * e_ij)
        self.gate_linear_src = nn.Linear(node_dim, edge_dim)
        self.gate_linear_dst = nn.Linear(node_dim, edge_dim)
        self.gate_linear_edge = nn.Linear(edge_dim, edge_dim)

        # Message transformation: W_msg * x_j
        self.msg_linear = nn.Linear(node_dim, node_dim)

        # Update mechanism for the node itself after aggregation
        self.update_linear = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, node_dim)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)
        Returns:
            Updated node features (N, node_dim)
        """
        # Propagate messages
        # x and edge_attr are passed to the message function
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Residual connection + Transformation + Norm
        out = x + self.dropout(self.update_linear(out))
        out = self.norm(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Computes the message for each edge (j -> i).
        x_i: Target node features (E, node_dim)
        x_j: Source node features (E, node_dim)
        edge_attr: Edge features (E, edge_dim)
        """
        # Compute Gate
        # We project node features to edge_dim to match edge_attr dimension for the gate
        gate_input = (
            self.gate_linear_src(x_i)
            + self.gate_linear_dst(x_j)
            + self.gate_linear_edge(edge_attr)
        )
        gate = torch.sigmoid(gate_input)

        # Compute raw message from source node
        msg = self.msg_linear(x_j)

        # Apply gate
        return gate * msg


class AngleAwareGNN(nn.Module):
    """
    Angle-Aware Gated Graph Network.
    Utilizes a dual-graph approach:
    1. Atom Graph: Nodes=Atoms, Edges=Bonds (Distances)
    2. Line Graph: Nodes=Bonds, Edges=Angles

    Message passing alternates between updating bonds (using angles) and updating atoms (using bonds).
    """

    def __init__(self):
        super().__init__()

        dim = Config.EMBEDDING_DIM

        # ---------------------------------------------------------------------
        # 1. Initial Embeddings
        # ---------------------------------------------------------------------
        # Atom embedding: Maps atomic numbers (mapped to indices 0-3) to vectors
        # We use a safe vocabulary size (e.g., 100) though we only have 4 types.
        self.atom_embedding = nn.Embedding(100, dim)

        # Distance RBF expansion and projection
        self.distance_rbf = GaussianRBF(0.0, Config.CUTOFF_RADIUS, Config.RBF_NUM_BINS)
        self.distance_proj = nn.Linear(Config.RBF_NUM_BINS, dim)

        # Angle RBF expansion and projection
        # Angles are in radians [0, Pi]
        self.angle_rbf = GaussianRBF(0.0, 3.14159, Config.RBF_NUM_BINS)
        self.angle_proj = nn.Linear(Config.RBF_NUM_BINS, dim)

        # ---------------------------------------------------------------------
        # 2. Interaction Blocks
        # ---------------------------------------------------------------------
        self.blocks = nn.ModuleList()
        for _ in range(Config.NUM_BLOCKS):
            # Line Graph Layer: Updates bond features.
            # Nodes are bonds (dim), Edges are angles (dim).
            line_layer = GatedGCNLayer(node_dim=dim, edge_dim=dim)

            # Atom Graph Layer: Updates atom features.
            # Nodes are atoms (dim), Edges are bonds (dim).
            atom_layer = GatedGCNLayer(node_dim=dim, edge_dim=dim)

            self.blocks.append(nn.ModuleDict({"line": line_layer, "atom": atom_layer}))

        # ---------------------------------------------------------------------
        # 3. Output Heads
        # ---------------------------------------------------------------------
        self.pool = global_mean_pool

        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )

    def forward(self, data):
        """
        Forward pass of the model.
        Args:
            data: PyG Data batch containing atom graph and line graph data.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr_rbf = data.edge_attr  # Raw distances

        line_edge_index = data.line_edge_index
        line_edge_attr_rbf = data.line_edge_attr  # Raw angles

        # ---------------------------------------------------------------------
        # Batching Correction for Line Graph
        # ---------------------------------------------------------------------
        # The line_edge_index provided by the Dataset is local to each graph.
        # When batched, we need to offset these indices because the "nodes" of the
        # line graph (which are the edges of the atom graph) are concatenated.
        # The offset for graph i is the cumulative number of atom edges in graphs 0...i-1.

        if hasattr(data, "num_atom_edges") and hasattr(data, "num_line_edges"):
            # Cite debug_lesson_2: Ensure num_bonds is flattened to 1D to match shifts calculation
            # This handles cases where Batch stacks scalars into (B, 1) or (B,)
            num_bonds = data.num_atom_edges.view(-1)
            num_angles = data.num_line_edges.view(-1)

            # Calculate cumulative offsets (0, N_b0, N_b0+N_b1, ...)
            # We want the offset for the i-th graph to be applied to all its line edges.
            shifts = torch.cumsum(num_bonds, dim=0) - num_bonds

            # If there are any angles in the batch
            if line_edge_index.size(1) > 0:
                # Repeat the offset for each angle in the corresponding graph
                offsets = torch.repeat_interleave(shifts, num_angles)
                # Apply offsets to both source and target indices of the line graph
                line_edge_index = line_edge_index + offsets.to(line_edge_index.device)

        # ---------------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------------
        # Embed Atoms
        h_atoms = self.atom_embedding(x)  # (N_atoms, dim)

        # Embed Bonds (Edges of Atom Graph -> Nodes of Line Graph)
        h_bonds = self.distance_rbf(edge_attr_rbf)  # (N_bonds, bins)
        h_bonds = self.distance_proj(h_bonds)  # (N_bonds, dim)

        # Embed Angles (Edges of Line Graph)
        h_angles = self.angle_rbf(line_edge_attr_rbf)  # (N_angles, bins)
        h_angles = self.angle_proj(h_angles)  # (N_angles, dim)

        # ---------------------------------------------------------------------
        # Message Passing Blocks
        # ---------------------------------------------------------------------
        for block in self.blocks:
            # 1. Line Graph Update
            # Update bond representations by passing messages across angles.
            # Nodes = h_bonds, Edges = h_angles, Connectivity = line_edge_index
            if line_edge_index.size(1) > 0:
                h_bonds = block["line"](h_bonds, line_edge_index, h_angles)

            # 2. Atom Graph Update
            # Update atom representations by passing messages across bonds.
            # Nodes = h_atoms, Edges = h_bonds (now updated with angular info), Connectivity = edge_index
            h_atoms = block["atom"](h_atoms, edge_index, h_bonds)

        # ---------------------------------------------------------------------
        # Readout and Prediction
        # ---------------------------------------------------------------------
        # Global mean pooling of atom features
        h_graph = self.pool(h_atoms, data.batch)

        # Predict targets
        pred_formation = self.head_formation(h_graph)
        pred_bandgap = self.head_bandgap(h_graph)

        # Concatenate predictions
        return torch.cat([pred_formation, pred_bandgap], dim=1)
