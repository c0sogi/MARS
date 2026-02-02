import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from library.config import Config
from library.geometry import GaussianSmearing, SphericalBasisLayer


class EmbeddingBlock(nn.Module):
    """
    Embeds atoms and edges.
    Atoms: Lookup table.
    Edges: RBF expansion + Linear projection.
    """

    def __init__(self, hidden_dim, num_rbf, max_atomic_num=10):
        super().__init__()
        self.atom_embedding = nn.Embedding(max_atomic_num + 1, hidden_dim)
        self.rbf = GaussianSmearing(
            start=0.0, stop=Config.CUTOFF, num_gaussians=num_rbf
        )
        self.edge_embedding = nn.Linear(num_rbf, hidden_dim)

    def forward(self, x, edge_attr):
        # x: [N] atomic numbers
        # edge_attr: [E] distances

        h = self.atom_embedding(x)  # [N, hidden_dim]

        rbf_feat = self.rbf(edge_attr)  # [E, num_rbf]
        m = self.edge_embedding(rbf_feat)  # [E, hidden_dim]

        return h, m, rbf_feat


class InteractionBlock(nn.Module):
    """
    Directional Message Passing Layer.
    Updates edge embeddings based on triplets and SBF features.
    Updates node embeddings based on aggregated edge embeddings.
    """

    def __init__(self, hidden_dim, num_rbf, num_sbf):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transformations for triplet aggregation
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sbf_proj = nn.Linear(num_sbf * num_rbf, hidden_dim)

        # Update function for edges
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Update function for nodes
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, h, m, rbf_feat, sbf_feat, triplets, edge_index):
        """
        h: [N, D] Node embeddings
        m: [E, D] Edge embeddings (directed)
        rbf_feat: [E, R] Radial basis features
        sbf_feat: [T, S] Spherical basis features for triplets
        triplets: [T, 2] Indices of (k->j, j->i) edges
        edge_index: [2, E] Graph connectivity
        """

        # 1. Triplet Message Passing (Update Edge Embeddings)
        if triplets.shape[0] > 0:
            # Gather features for incoming edges (k->j)
            # triplets[:, 0] is the index of edge k->j
            m_kj = m[triplets[:, 0]]  # [T, D]

            # Project incoming edges and SBF features
            # Element-wise interaction between geometric features and edge features
            msg_triplet = self.k_proj(m_kj) * self.sbf_proj(sbf_feat)  # [T, D]

            # Aggregate messages to outgoing edges (j->i)
            # triplets[:, 1] is the index of edge j->i
            # We sum all messages contributing to the same outgoing edge
            m_aggr = scatter_add(
                msg_triplet, triplets[:, 1], dim=0, dim_size=m.size(0)
            )  # [E, D]
        else:
            m_aggr = torch.zeros_like(m)

        # Combine: Current Edge + Aggregated Triplet Info + Radial Info
        # We concat [m, m_aggr, rbf] to preserve radial info at every step
        m_new_input = torch.cat([m, m_aggr, rbf_feat], dim=-1)
        m_new = self.edge_update_mlp(m_new_input) + m  # Residual connection

        # 2. Node Update
        # Aggregate incoming edges to nodes
        # edge_index[1] is the target node i for edge j->i
        node_aggr = scatter_add(
            m_new, edge_index[1], dim=0, dim_size=h.size(0)
        )  # [N, D]

        h_new_input = torch.cat([h, node_aggr], dim=-1)
        h_new = self.node_update_mlp(h_new_input) + h  # Residual connection

        return h_new, m_new


class MultiHeadReadout(nn.Module):
    """
    Predicts scalar coupling constant using type-specific heads.
    """

    def __init__(self, hidden_dim, num_types=8):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_types)
            ]
        )

    def forward(self, h, coupling_index, coupling_type):
        """
        h: [N, D] Node embeddings
        coupling_index: [2, C] Atom pairs
        coupling_type: [C] Integer type indices
        """
        num_couplings = coupling_type.size(0)
        if num_couplings == 0:
            return torch.zeros(0, device=h.device)

        # Prepare output tensor
        preds = torch.zeros(num_couplings, 1, device=h.device)

        # Iterate over each type and apply specific head
        unique_types = torch.unique(coupling_type)
        for t in unique_types:
            t = int(t)
            mask = coupling_type == t

            # Get indices for this type
            idx_0 = coupling_index[0, mask]
            idx_1 = coupling_index[1, mask]

            # Concatenate node features
            h_pair = torch.cat([h[idx_0], h[idx_1]], dim=-1)

            # Predict
            out = self.heads[t](h_pair)

            # Assign back
            preds[mask] = out

        return preds.squeeze(-1)


class PhysicsAwareNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Hyperparameters
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_rbf = Config.NUM_RBF
        self.num_sbf = Config.NUM_SBF
        self.cutoff = Config.CUTOFF

        # 1. Basis Layers
        self.sbf_layer = SphericalBasisLayer(self.num_sbf, self.num_rbf, self.cutoff)

        # 2. Embedding
        self.embedding = EmbeddingBlock(
            self.hidden_dim, self.num_rbf, Config.MAX_ATOMIC_NUM
        )

        # 3. Interaction Blocks
        self.layers = nn.ModuleList(
            [
                InteractionBlock(self.hidden_dim, self.num_rbf, self.num_sbf)
                for _ in range(self.num_layers)
            ]
        )

        # 4. Auxiliary Heads (Physics Supervision)
        # Shielding: 9 components (XX, YX, ..., ZZ)
        self.shielding_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 9),
        )
        # Charge: 1 component
        self.charge_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # 5. Main Task Readout
        self.readout = MultiHeadReadout(self.hidden_dim, Config.NUM_COUPLING_TYPES)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        triplets,
        triplet_attr,
        coupling_index,
        coupling_type,
        **kwargs
    ):
        """
        Forward pass of the network.

        Args:
            x: [N] Atomic numbers
            edge_index: [2, E]
            edge_attr: [E] Distances
            triplets: [T, 2] Indices of edge pairs for angles
            triplet_attr: [T] Angles (radians)
            coupling_index: [2, C] Pairs to predict
            coupling_type: [C] Type indices
        """

        # --- Precompute Geometric Features ---

        # SBF requires distance of the incoming edge (k->j) and the angle
        if triplets.shape[0] > 0:
            # triplets[:, 0] is index of edge k->j
            dist_kj = edge_attr[triplets[:, 0]]
            sbf_feat = self.sbf_layer(dist_kj, triplet_attr)
        else:
            sbf_feat = torch.zeros((0, self.num_sbf * self.num_rbf), device=x.device)

        # --- Embedding ---
        h, m, rbf_feat = self.embedding(x, edge_attr)

        # --- Message Passing ---
        for layer in self.layers:
            h, m = layer(h, m, rbf_feat, sbf_feat, triplets, edge_index)

        # --- Auxiliary Predictions ---
        # Predict physics properties for every node
        shielding_pred = self.shielding_head(h)
        charge_pred = self.charge_head(h)

        # --- Main Task Prediction ---
        coupling_pred = self.readout(h, coupling_index, coupling_type)

        return coupling_pred, shielding_pred, charge_pred
