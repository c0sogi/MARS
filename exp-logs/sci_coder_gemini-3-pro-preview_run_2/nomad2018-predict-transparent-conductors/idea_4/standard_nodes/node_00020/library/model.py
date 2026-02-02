import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, CGConv
from torch_geometric.data import Batch
from library.config import Config
from library.utils import GaussianRBF, Standardizer, compute_rmsle


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    Uses the CGConv layer from PyTorch Geometric, which implements the
    continuous filter convolution update:
    x_i' = x_i + sum_{j} sigma(z_ij W_f + b_f) * g(z_ij W_s + b_s)
    where z_ij is the edge feature vector (RBF expanded distance).

    This architecture explicitly models atomic interactions based on distance
    and has a strong inductive bias for crystal systems.
    (Cite solution_lesson_node_00011, solution_lesson_node_00018)
    """

    def __init__(self):
        super().__init__()

        dim = Config.EMBEDDING_DIM

        # 1. Initial Embeddings
        self.atom_embedding = nn.Embedding(100, dim)
        self.distance_rbf = GaussianRBF(0.0, Config.CUTOFF_RADIUS, Config.RBF_NUM_BINS)

        # Project RBF to embedding dimension for CGConv
        # CGConv expects edge_dim to match node_dim or be specified.
        # Here we project to 'dim' to match node features.
        self.edge_embedding = nn.Linear(Config.RBF_NUM_BINS, dim)

        # 2. Interaction Blocks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(Config.NUM_BLOCKS):
            # CGConv(channels, dim, ...)
            # channels: node feature dim (in/out)
            # dim: edge feature dim
            self.convs.append(
                CGConv(channels=dim, dim=dim, aggr="add", batch_norm=True)
            )
            # Although CGConv has internal batch_norm option, explicit norm is often safer
            # But here we stick to the layer definition.
            # We add an activation after the residual block if needed, but CGCNN usually
            # uses the result of the convolution directly.
            # Let's add a non-linearity after the block.

        # 3. Output Heads
        self.pool = global_mean_pool

        self.head_formation = nn.Sequential(
            nn.Linear(dim, dim), nn.Softplus(), nn.Linear(dim, 1)
        )

        self.head_bandgap = nn.Sequential(
            nn.Linear(dim, dim), nn.Softplus(), nn.Linear(dim, 1)
        )

        # Dropout for regularization (Cite solution_lesson_node_00003)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Embed Atoms
        h = self.atom_embedding(x)  # (N, dim)

        # Embed Edges
        edge_feat = self.distance_rbf(edge_attr)  # (E, bins)
        edge_feat = self.edge_embedding(edge_feat)  # (E, dim)

        # Message Passing
        for conv in self.convs:
            # CGConv applies the update x_new = x + conv(x, edge_index, edge_attr)
            # It handles the residual internally if we don't overwrite x directly?
            # No, PyG layers usually return the transformed feature.
            # CGConv source: out = prop(...) ... return x + out (if residual=True, default is False in source?)
            # Checking PyG docs: CGConv does NOT have a residual argument in constructor,
            # but the formula implies it adds to x_i.
            # Actually, standard implementation: x_new = conv(x, ...).
            # We need to add residual manually if the layer doesn't.
            # Let's assume standard PyG usage: h = conv(h, ...).
            # Wait, CGCNN paper has residual.
            # PyG CGConv implementation:
            #   z = cat([x_i, x_j, edge_attr], dim=-1)
            #   out = sigmoid(z W_f) * softplus(z W_s)
            #   return scatter(out, ...)
            # It does NOT add x_i automatically.

            h_update = conv(h, edge_index, edge_feat)
            h = h + h_update
            # Apply non-linearity? CGCNN usually relies on the internal gates.
            # But standard ResNet practice suggests a non-linearity after addition?
            # Original CGCNN paper: v_i^(t+1) = v_i^t + sum(...)
            # Then passed to next layer.
            # We can apply a Softplus activation here as in some implementations
            h = F.softplus(h)
            h = self.dropout(h)

        # Readout
        h_graph = self.pool(h, data.batch)

        # Predict
        pred_formation = self.head_formation(h_graph)
        pred_bandgap = self.head_bandgap(h_graph)

        return torch.cat([pred_formation, pred_bandgap], dim=1)
