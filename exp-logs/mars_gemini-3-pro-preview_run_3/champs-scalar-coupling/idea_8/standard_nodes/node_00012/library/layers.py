import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from library.config import Config
from library.utils import GaussianSmearing


class CFConv(MessagePassing):
    """
    Continuous Filter Convolution Layer.
    Generates dynamic filters from edge attributes using an MLP.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__(aggr="add")
        # Filter Generator: Transforms edge attributes (RBFs + embeddings) into filter weights
        self.filter_network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
        )
        # Transformation for neighbor node features before aggregation
        self.lin_neighbor = nn.Linear(node_dim, out_dim)
        # Update network applied after aggregation
        self.lin_update = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Generate dynamic filters
        # edge_attr: [E, edge_dim] -> W: [E, out_dim]
        W = self.filter_network(edge_attr)

        # Propagate messages
        # x: [N, node_dim]
        out = self.propagate(edge_index, x=x, W=W)

        # Apply final update
        out = self.lin_update(out)
        return out

    def message(self, x_j, W):
        # x_j: [E, node_dim] (Features of neighbor nodes)
        # W: [E, out_dim] (Generated filters)
        # Element-wise multiplication (SchNet-style interaction)
        return self.lin_neighbor(x_j) * W


class DualInteractionBlock(nn.Module):
    """
    Interaction block that updates both Edge (Line Graph) and Atom (Atom Graph) representations.
    """

    def __init__(self, hidden_channels, num_rbf_dist, num_rbf_angle):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Normalization layers for stability
        self.norm_atom = nn.LayerNorm(hidden_channels)
        self.norm_edge = nn.LayerNorm(hidden_channels)

        # --- Line Graph Update (Edge-to-Edge) ---
        # Updates edge embeddings based on angular relations between bonds.
        # The 'nodes' in this graph are the edges of the molecule.
        # The 'edges' in this graph represent angles.
        self.conv_line = CFConv(
            node_dim=hidden_channels,
            edge_dim=num_rbf_angle,
            hidden_dim=hidden_channels,
            out_dim=hidden_channels,
        )

        # --- Atom Graph Update (Node-to-Node) ---
        # Updates atom embeddings based on spatial distance AND evolved edge embeddings.
        # The filter is conditioned on both distance RBF and the current edge state.
        self.conv_atom = CFConv(
            node_dim=hidden_channels,
            edge_dim=num_rbf_dist + hidden_channels,  # Concatenated features
            hidden_dim=hidden_channels,
            out_dim=hidden_channels,
        )

    def forward(
        self, x, edge_embeddings, edge_index, dist_rbf, line_edge_index, angle_rbf
    ):
        # 1. Line Graph Step: Update Edge Embeddings
        # Residual connection + Norm
        h_edge_in = self.norm_edge(edge_embeddings)

        # Convolution on Line Graph
        # Input 'x' is edge_embeddings. Input 'edge_attr' is angle_rbf.
        h_edge_out = self.conv_line(h_edge_in, line_edge_index, angle_rbf)

        # Residual update
        edge_embeddings = edge_embeddings + h_edge_out

        # 2. Atom Graph Step: Update Atom Embeddings
        # Residual connection + Norm
        h_atom_in = self.norm_atom(x)

        # Construct composite edge attributes for the atom filter
        # We inject the updated edge_embeddings into the atom interaction
        # [E, num_rbf_dist] cat [E, hidden] -> [E, num_rbf_dist + hidden]
        composite_edge_attr = torch.cat([dist_rbf, edge_embeddings], dim=-1)

        # Convolution on Atom Graph
        h_atom_out = self.conv_atom(h_atom_in, edge_index, composite_edge_attr)

        # Residual update
        x = x + h_atom_out

        return x, edge_embeddings


class SDG_CFC(nn.Module):
    """
    Stabilized Dual-Graph Network with Continuous Filter Convolutions.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.hidden_channels = Config.HIDDEN_CHANNELS
        self.num_layers = Config.NUM_LAYERS
        self.num_rbf_dist = Config.NUM_RBF_DISTANCE
        self.num_rbf_angle = Config.NUM_RBF_ANGLE

        # 1. Initial Embeddings
        self.embedding_atom = nn.Embedding(Config.NUM_ATOM_TYPES, self.hidden_channels)

        # Initialize edge embedding from distance RBF
        self.embedding_edge = nn.Sequential(
            nn.Linear(self.num_rbf_dist, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # RBF Expanders
        # Distance RBF (0 to Cutoff)
        self.rbf_dist = GaussianSmearing(
            start=0.0, stop=Config.CUTOFF_RADIUS, num_gaussians=self.num_rbf_dist
        )
        # Angle RBF (Cosine -1 to 1)
        self.rbf_angle = GaussianSmearing(
            start=-1.0, stop=1.0, num_gaussians=self.num_rbf_angle
        )

        # 2. Backbone (Dual Interaction Blocks)
        self.blocks = nn.ModuleList(
            [
                DualInteractionBlock(
                    self.hidden_channels, self.num_rbf_dist, self.num_rbf_angle
                )
                for _ in range(self.num_layers)
            ]
        )

        # 3. Readout Heads
        # Input: Atom_i + Atom_j + Edge_ij
        readout_input_dim = 3 * self.hidden_channels

        # Type-Specific Heads (one for each coupling type)
        self.heads = nn.ModuleList()
        for _ in range(Config.NUM_HEADS):
            self.heads.append(
                nn.Sequential(
                    nn.Linear(readout_input_dim, self.hidden_channels),
                    nn.SiLU(),
                    nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                    nn.SiLU(),
                    nn.Linear(self.hidden_channels // 2, 1),
                )
            )

        # 4. Auxiliary Heads (Physics Regularization)
        # Magnetic Shielding (9 components)
        self.head_shielding = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_channels // 2, 9),
        )

        # Mulliken Charge (1 component)
        self.head_charge = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_channels // 2, 1),
        )

    def get_edge_embeddings_for_couplings(
        self, edge_embeddings, edge_index, coupling_index, num_nodes
    ):
        """
        Retrieves edge embeddings for the specified coupling pairs using a sparse-to-dense trick.
        This handles the mapping from the coupling pair (u, v) to the specific edge index in the graph.
        """
        device = edge_embeddings.device

        # Create a sparse tensor representing the edge indices
        # Values are indices + 1 (to distinguish from missing edges/zeros)
        E = edge_index.size(1)
        indices = edge_index
        values = torch.arange(1, E + 1, device=device)

        # Create sparse tensor [N, N]
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

        # Convert to dense for efficient lookup
        # Note: For molecular graphs in batches, N is typically < 5000, so N^2 is manageable on GPU.
        adj_dense = adj.to_dense()

        # Query the dense matrix at the coupling coordinates
        rows = coupling_index[0]
        cols = coupling_index[1]

        found_indices = adj_dense[rows, cols]  # [K]

        # Mask for valid edges (found_indices > 0)
        mask = found_indices > 0
        valid_indices = found_indices[mask] - 1  # Convert back to 0-based index

        # Prepare output container
        K = coupling_index.size(1)
        out_embeddings = torch.zeros(K, self.hidden_channels, device=device)

        # Fill found embeddings
        if mask.any():
            out_embeddings[mask] = edge_embeddings[valid_indices]

        return out_embeddings

    def forward(self, data):
        # Unpack Data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr  # Distances [E, 1]
        line_edge_index = data.line_edge_index
        line_edge_attr = data.line_edge_attr  # Angles [L, 1]

        # 1. Feature Expansion (RBF)
        dist_rbf = self.rbf_dist(edge_attr)
        angle_rbf = self.rbf_angle(line_edge_attr)

        # 2. Initialization
        h_atom = self.embedding_atom(x)
        h_edge = self.embedding_edge(dist_rbf)

        # 3. Message Passing Backbone
        for block in self.blocks:
            h_atom, h_edge = block(
                h_atom, h_edge, edge_index, dist_rbf, line_edge_index, angle_rbf
            )

        # 4. Auxiliary Predictions (Node-level)
        pred_shielding = self.head_shielding(h_atom)
        pred_charge = self.head_charge(h_atom)

        # 5. Coupling Prediction (Pair-level)
        coupling_index = data.coupling_index  # [2, K]
        coupling_type = data.coupling_type  # [K]

        # Gather atom features for the pairs
        idx_0 = coupling_index[0]
        idx_1 = coupling_index[1]

        h_0 = h_atom[idx_0]
        h_1 = h_atom[idx_1]

        # Gather edge features for the pairs (Edge Injection)
        # data.num_nodes represents total nodes in the current batch
        h_e_pair = self.get_edge_embeddings_for_couplings(
            h_edge, edge_index, coupling_index, data.num_nodes
        )

        # Concatenate features: [h_i || h_j || h_e_ij]
        z = torch.cat([h_0, h_1, h_e_pair], dim=-1)

        # Route to Type-Specific Heads
        pred_coupling = torch.zeros(z.size(0), 1, device=z.device)

        for t_idx in range(Config.NUM_HEADS):
            # Create mask for current coupling type
            mask = coupling_type == t_idx
            if mask.any():
                # Select subset
                z_subset = z[mask]
                # Predict
                out_subset = self.heads[t_idx](z_subset)
                # Scatter back results
                pred_coupling[mask] = out_subset

        return pred_coupling.squeeze(-1), pred_shielding, pred_charge
