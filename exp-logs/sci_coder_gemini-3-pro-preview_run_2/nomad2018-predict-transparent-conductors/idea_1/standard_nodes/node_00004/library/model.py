import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class RBFExpansion(nn.Module):
    """
    Expands distances into a vector of radial basis functions (Gaussian).
    """

    def __init__(self, vmin=0, vmax=8.0, bins=40, length=None):
        super(RBFExpansion, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.center = torch.linspace(vmin, vmax, bins)
        # Gamma controls the width of the Gaussian.
        # A common heuristic is 1 / (step_size)^2
        self.gamma = 1.0 / ((vmax - vmin) / bins) ** 2

        if length is None:
            self.length = bins
        else:
            self.length = length

        self.register_buffer("centers", self.center)

    def forward(self, distance):
        """
        Args:
            distance (Tensor): [n_edges] tensor of distances.
        Returns:
            Tensor: [n_edges, bins] tensor of expanded distances.
        """
        # distance: [E] -> [E, 1]
        # centers: [bins] -> [1, bins]
        # result: [E, bins]
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2
        )


class CGCNNLayer(nn.Module):
    """
    Convolutional layer for Crystal Graph Convolutional Neural Network.
    Uses a gated activation mechanism applied to concatenated node and edge features.
    """

    def __init__(self, atom_fea_len, edge_fea_len):
        super(CGCNNLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.edge_fea_len = edge_fea_len

        # Input dimension: atom_fea (i) + atom_fea (j) + edge_fea (ij)
        self.z_dim = 2 * atom_fea_len + edge_fea_len

        # Linear layers for the gated mechanism
        self.sigmoid_linear = nn.Linear(self.z_dim, atom_fea_len)
        self.softplus_linear = nn.Linear(self.z_dim, atom_fea_len)

        # Batch normalization for stability
        self.bn_softplus = nn.BatchNorm1d(atom_fea_len)
        self.bn_sigmoid = nn.BatchNorm1d(atom_fea_len)

    def forward(self, atom_in_fea, edge_index, edge_fea):
        """
        Args:
            atom_in_fea (Tensor): [N, atom_fea_len] Node features.
            edge_index (Tensor): [2, E] Edge indices (src, dst).
            edge_fea (Tensor): [E, edge_fea_len] Edge features.
        Returns:
            Tensor: [N, atom_fea_len] Updated node features.
        """
        # Get source and target node features
        # edge_index[0] is source, edge_index[1] is destination
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        atom_src = atom_in_fea[src_idx]  # [E, F]
        atom_dst = atom_in_fea[dst_idx]  # [E, F]

        # Concatenate features for messages: z_ij = [v_i, v_j, u_ij]
        z = torch.cat([atom_src, atom_dst, edge_fea], dim=1)  # [E, 2F + F_edge]

        # Compute gated activation
        # Filter: Softplus(Linear(z))
        # Gate: Sigmoid(Linear(z))
        # Message = Filter * Gate
        filter_out = self.bn_softplus(F.softplus(self.softplus_linear(z)))
        gate_out = self.bn_sigmoid(torch.sigmoid(self.sigmoid_linear(z)))

        message = filter_out * gate_out  # [E, F]

        # Aggregate messages to destination nodes using scatter_add
        # dst_idx is the index to scatter into
        # dim=0 because we are aggregating rows
        # dim_size is number of atoms to ensure output shape matches atom_in_fea
        aggr_message = scatter(
            message, dst_idx, dim=0, dim_size=atom_in_fea.size(0), reduce="add"
        )

        # Residual connection
        atom_out_fea = atom_in_fea + aggr_message

        return atom_out_fea


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.
    Predicts formation energy and bandgap energy.
    """

    def __init__(
        self,
        orig_atom_fea_len=4,  # 4 types of atoms: Al, Ga, In, O
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        n_targets=2,
        radius=5.0,
        n_rbf=40,
        dropout=0.2,
    ):
        super(CGCNN, self).__init__()

        # Embedding for atom types (0:O, 1:Al, 2:Ga, 3:In)
        self.embedding = nn.Embedding(orig_atom_fea_len, atom_fea_len)

        # RBF Expansion for edge distances
        self.rbf_expansion = RBFExpansion(vmin=0, vmax=radius, bins=n_rbf)

        # Stack of Convolution layers
        self.convs = nn.ModuleList(
            [
                CGCNNLayer(atom_fea_len=atom_fea_len, edge_fea_len=n_rbf)
                for _ in range(n_conv)
            ]
        )

        # Global pooling to Output MLP
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.bn_hidden = nn.BatchNorm1d(h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(h_fea_len) for _ in range(n_h - 1)]
            )
        else:
            self.fcs = nn.ModuleList([])
            self.bns = nn.ModuleList([])

        self.fc_out = nn.Linear(h_fea_len, n_targets)
        self.dropout = nn.Dropout(p=dropout)

        self.n_h = n_h

    def forward(self, atom_fea, edge_index, edge_dist, batch_index):
        """
        Args:
            atom_fea (Tensor): [N] Atom type indices.
            edge_index (Tensor): [2, E] Edge connectivity.
            edge_dist (Tensor): [E] Edge distances.
            batch_index (Tensor): [N] Batch index for each node (for pooling).
        Returns:
            Tensor: [B, n_targets] Predictions.
        """
        # 1. Node Embedding
        x = self.embedding(atom_fea)  # [N, atom_fea_len]

        # 2. Edge Embedding (RBF)
        edge_fea = self.rbf_expansion(edge_dist)  # [E, n_rbf]

        # 3. Convolution Layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_fea)

        # 4. Global Pooling
        # Aggregate node features into graph features
        # We use mean pooling to be invariant to crystal size in a stable way
        crys_fea = scatter(x, batch_index, dim=0, reduce="mean")  # [B, atom_fea_len]

        # 5. MLP
        crys_fea = F.softplus(self.conv_to_fc(crys_fea))
        crys_fea = self.bn_hidden(crys_fea)
        crys_fea = self.dropout(crys_fea)

        for i in range(self.n_h - 1):
            crys_fea = F.softplus(self.fcs[i](crys_fea))
            crys_fea = self.bns[i](crys_fea)
            crys_fea = self.dropout(crys_fea)

        out = self.fc_out(crys_fea)

        return out
