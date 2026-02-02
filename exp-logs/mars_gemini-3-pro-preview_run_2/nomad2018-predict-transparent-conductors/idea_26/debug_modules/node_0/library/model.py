import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from library.config import Config
from library.utils import RBFExpansion


class RAGLUInteractionBlock(nn.Module):
    """
    Receiver-Aware Gated Linear Unit Interaction Block.

    Performs message passing using a triplet input (target, source, edge) fed into a GLU,
    followed by aggregation and a residual connection weighted by a learnable scalar.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Input: Concatenation of Target Node (h_i), Source Node (h_j), and Edge Embedding (e_ij)
        # Dimension: 3 * hidden_dim
        # Output: 2 * hidden_dim (for GLU splitting)
        self.glu_linear = nn.Linear(hidden_dim * 3, hidden_dim * 2)

        # Learnable scalar residual weight, initialized to 0
        self.epsilon = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.Softplus()

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Shared edge embeddings [num_edges, hidden_dim]
        """
        src, dst = edge_index

        # Receiver-Aware Triplet Construction
        x_j = x[src]  # Source nodes
        x_i = x[dst]  # Target nodes

        # z_ij = [h_i || h_j || e_ij]
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [num_edges, 3 * hidden_dim]

        # Gated Linear Unit Mechanism
        glu_out = self.glu_linear(z_ij)  # [num_edges, 2 * hidden_dim]
        a, b = glu_out.chunk(2, dim=-1)
        m_ij = a * torch.sigmoid(b)  # [num_edges, hidden_dim]

        # Aggregation
        aggr_out = scatter_add(
            m_ij, dst, dim=0, dim_size=x.size(0)
        )  # [num_nodes, hidden_dim]

        # Residual Connection with Learnable Scalar
        # h_{l+1} = Softplus(BatchNorm(Agg(m_ij) + (1 + epsilon) * h_l))
        out = aggr_out + (1.0 + self.epsilon) * x
        out = self.bn(out)
        out = self.activation(out)

        return out


class RAGLUNet(nn.Module):
    """
    Receiver-Aware Gated Linear Unit Network (RA-GLU-Net).

    Features:
    - Radius Graph with PBC (handled by data loader)
    - Node Embeddings based on atomic number
    - Shared Linear Projection for RBF edge features
    - Stack of RAGLUInteractionBlocks
    - Global Mean Pooling
    - Decoupled MLP heads for multi-target prediction
    """

    def __init__(self, config=Config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS

        # 1. Node Embedding
        self.node_embedding = nn.Embedding(
            config.MAX_ATOMIC_NUMBER + 1, self.hidden_dim
        )

        # 2. Edge Embedding & Shared Projection
        # Expands distances using RBF, then projects to hidden_dim once
        self.rbf = RBFExpansion(vmin=0.0, vmax=config.CUTOFF, bins=config.NUM_RBF)
        self.edge_proj = nn.Linear(config.NUM_RBF, self.hidden_dim)

        # 3. Interaction Blocks
        self.blocks = nn.ModuleList(
            [RAGLUInteractionBlock(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # 4. Decoupled Readout Heads
        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric style batch object containing:
                - x: Atomic numbers [num_nodes]
                - edge_index: [2, num_edges]
                - edge_attr: Distances [num_edges, 1]
                - batch: Batch indices [num_nodes]
        """
        x = data["x"]
        edge_index = data["edge_index"]
        edge_attr_dist = data["edge_attr"]
        batch = data["batch"]

        # Initial Node Embeddings
        h = self.node_embedding(x)  # [num_nodes, hidden_dim]

        # Shared Edge Embeddings
        # Expand distances and project once
        rbf_feat = self.rbf(edge_attr_dist.squeeze(-1))  # [num_edges, num_rbf]
        edge_emb = self.edge_proj(rbf_feat)  # [num_edges, hidden_dim]

        # Message Passing
        for block in self.blocks:
            h = block(h, edge_index, edge_emb)

        # Global Pooling
        # Global Mean Pooling aligns with intensive properties
        h_pool = scatter_mean(h, batch, dim=0)  # [batch_size, hidden_dim]

        # Prediction Heads
        pred_formation = self.head_formation(h_pool)
        pred_bandgap = self.head_bandgap(h_pool)

        # Concatenate predictions [batch_size, 2]
        return torch.cat([pred_formation, pred_bandgap], dim=1)
