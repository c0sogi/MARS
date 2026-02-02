import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention
from library.config import Config


class CGCNNConv(MessagePassing):
    """
    Gated Graph Convolution Layer as used in CGCNN.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super(CGCNNConv, self).__init__(aggr="add", flow="target_to_source")
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        # Weight matrix for concatenation of [v_i, v_j, u_ij]
        # Input dim: atom_fea_len (vi) + atom_fea_len (vj) + nbr_fea_len (uij)
        # Output dim: 2 * atom_fea_len (filter + core)
        self.linear = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.bn = nn.BatchNorm1d(2 * atom_fea_len)

        # Batch norm after aggregation
        self.bn_out = nn.BatchNorm1d(atom_fea_len)

    def forward(self, x, edge_index, edge_attr):
        """
        x: Node features (N, atom_fea_len)
        edge_index: Graph connectivity (2, E)
        edge_attr: Edge features (E, nbr_fea_len)
        """
        # Save x for residual connection
        x_res = x

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply batch norm to the aggregated messages
        out = self.bn_out(out)

        # Residual connection + Softplus activation
        return F.softplus(out + x_res)

    def message(self, x_i, x_j, edge_attr):
        """
        Construct message: z_ij = [x_i, x_j, edge_attr]
        Apply linear transformation and gating.
        """
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        z = self.linear(z)
        z = self.bn(z)

        # Split into filter (sigmoid) and core (softplus) parts
        filter_weight, core_weight = z.split(self.atom_fea_len, dim=1)

        # Gated activation: sigmoid(filter) * softplus(core)
        return torch.sigmoid(filter_weight) * F.softplus(core_weight)


class CompositionAwareCGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network with:
    1. Gated Graph Convolutions (Backbone)
    2. Global Attention Pooling (Aggregation)
    3. Late Fusion of Global Features (Lattice/Composition)
    4. Dual-Head Output (Formation Energy & Bandgap)
    """

    def __init__(self, config=Config):
        super(CompositionAwareCGCNN, self).__init__()

        self.atom_fea_len = config.ATOM_FEA_LEN
        self.nbr_fea_len = config.RBF_N_BINS
        self.n_conv = config.N_CONV
        self.global_fea_len = config.GLOBAL_FEA_LEN
        self.dropout_rate = config.DROPOUT

        # 1. Node Embedding
        # Maps one-hot atom vectors to continuous embedding
        self.embedding = nn.Linear(len(config.ATOM_TYPES), self.atom_fea_len)

        # 2. CGCNN Backbone
        self.convs = nn.ModuleList(
            [CGCNNConv(self.atom_fea_len, self.nbr_fea_len) for _ in range(self.n_conv)]
        )

        # 3. Global Attention Pooling
        # Computes attention weights for weighted sum aggregation
        gate_nn = nn.Sequential(
            nn.Linear(self.atom_fea_len, 32), nn.Softplus(), nn.Linear(32, 1)
        )
        self.pooling = GlobalAttention(gate_nn=gate_nn, nn=None)

        # 4. Global Feature Processing MLP
        # Processes lattice parameters and composition
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_fea_len, 64),
            nn.BatchNorm1d(64),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Softplus(),
        )

        # 5. Fusion Dimension
        fusion_dim = self.atom_fea_len + 64

        # 6. Readout Heads
        # Head 1: Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1),
        )

        # Head 2: Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: PyG DataBatch object containing:
                - x: Node features (Batch*N, n_atom_types)
                - edge_index: Graph connectivity (2, E)
                - edge_attr: Edge features (E, nbr_fea_len)
                - global_feat: Global features (Batch, 1, global_fea_len) or (Batch, global_fea_len)
                - batch: Batch index for nodes (Batch*N)

        Returns:
            out: Prediction tensor (Batch, 2)
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        global_feat = data.global_feat

        # Fix global_feat shape if necessary (Batch, 1, 10) -> (Batch, 10)
        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        # 1. Embed Nodes
        x = self.embedding(x)

        # 2. Graph Convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # 3. Global Attention Pooling
        # Aggregates node features into a graph vector
        graph_embedding = self.pooling(x, batch)

        # 4. Process Global Features
        global_embedding = self.global_mlp(global_feat)

        # 5. Late Fusion
        fused_embedding = torch.cat([graph_embedding, global_embedding], dim=1)

        # 6. Dual Head Prediction
        pred_formation = self.head_formation(fused_embedding)
        pred_bandgap = self.head_bandgap(fused_embedding)

        # Concatenate outputs
        return torch.cat([pred_formation, pred_bandgap], dim=1)
