import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from library.config import Config


class DistanceWeightedGCN(nn.Module):
    """
    A Distance-Weighted Graph Convolutional Network for predicting scalar coupling constants.

    Architecture:
    1. Atom & Type Embeddings: Embeds atom types and coupling types into dense vectors.
    2. GCN Layers: Multiple GCN layers where message passing is weighted by inverse Euclidean distance.
    3. Readout Head: Extracts embeddings for the interacting atom pair, concatenates them with
       their spatial distance and coupling type embedding, and passes through an MLP.
    """

    def __init__(self):
        super(DistanceWeightedGCN, self).__init__()

        # Hyperparameters from Config
        self.atom_embed_dim = Config.ATOM_EMBED_DIM
        self.type_embed_dim = Config.TYPE_EMBED_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_GCN_LAYERS
        self.dropout_rate = Config.DROPOUT
        self.mlp_hidden_dim = Config.MLP_HIDDEN_DIM

        # 1. Embeddings
        # Atom Embedding (e.g., H, C, N...)
        self.atom_embedding = nn.Embedding(len(Config.ATOM_MAP), self.atom_embed_dim)

        # Coupling Type Embedding (e.g., 1JHC, 2JHH...)
        self.type_embedding = nn.Embedding(len(Config.TYPE_MAP), self.type_embed_dim)

        # 2. Graph Convolutional Layers
        self.convs = nn.ModuleList()

        # First layer: Atom Embedding -> Hidden Dim
        self.convs.append(GCNConv(self.atom_embed_dim, self.hidden_dim))

        # Subsequent layers: Hidden Dim -> Hidden Dim
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # 3. Readout / Prediction Head
        # Input features: Node_i (Hidden) + Node_j (Hidden) + Distance (1) + Type (Type_Embed)
        input_dim = (self.hidden_dim * 2) + 1 + self.type_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (torch_geometric.data.Data or Batch): Input graph data containing:
                - x: Atom type indices [Num_Nodes]
                - edge_index: Graph connectivity [2, Num_Edges]
                - edge_attr: Inverse distances [Num_Edges, 1]
                - pos: Cartesian coordinates [Num_Nodes, 3]
                - couple_index: Indices of atom pairs to predict [2, Num_Couples]
                - couple_type: Coupling type indices [Num_Couples]
                - batch (optional): Graph assignment for nodes [Num_Nodes]

        Returns:
            torch.Tensor: Predicted scalar coupling constants [Num_Couples, 1]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # --- 1. Graph Representation Learning ---

        # Embed atom types
        h = self.atom_embedding(x)

        # Prepare edge weights (Inverse Distance)
        # GCNConv expects edge_weight to be 1D [Num_Edges]
        edge_weight = edge_attr.squeeze(-1)

        # Apply GCN Layers
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # --- 2. Pairwise Feature Extraction ---

        # The 'couple_index' contains local atom indices (0 to N_atoms_in_molecule - 1).
        # When processing a Batch of graphs, we need to adjust these indices to point
        # to the correct nodes in the concatenated 'h' and 'pos' tensors.

        if isinstance(data, Batch):
            # data.ptr contains the start index of nodes for each graph in the batch
            # Shape: [Num_Graphs + 1], e.g., [0, N1, N1+N2, ...]
            node_offsets = data.ptr[:-1]

            # We need to broadcast these offsets to match the number of couples in each graph.
            # Using to_data_list() is a robust way to retrieve the number of couples per graph
            # without relying on internal PyG slice dictionaries.
            data_list = data.to_data_list()
            couple_counts = [d.couple_index.size(1) for d in data_list]
            couple_counts_tensor = torch.tensor(couple_counts, device=x.device)

            # Create a shift vector: [Offset_G1...Offset_G1, Offset_G2...Offset_G2, ...]
            shifts = torch.repeat_interleave(node_offsets, couple_counts_tensor)

            # Apply shifts to local indices to get global batch indices
            idx0 = data.couple_index[0] + shifts
            idx1 = data.couple_index[1] + shifts
        else:
            # Single graph inference
            idx0 = data.couple_index[0]
            idx1 = data.couple_index[1]

        # Gather Node Embeddings for the pairs
        h0 = h[idx0]
        h1 = h[idx1]

        # Calculate Euclidean Distance for the pairs
        # We calculate this on-the-fly using the global positions
        pos0 = data.pos[idx0]
        pos1 = data.pos[idx1]
        dist = torch.norm(pos0 - pos1, p=2, dim=-1, keepdim=True)

        # Get Coupling Type Embedding
        type_emb = self.type_embedding(data.couple_type)

        # --- 3. Prediction ---

        # Concatenate all features: [Node0_Emb, Node1_Emb, Distance, Type_Emb]
        out = torch.cat([h0, h1, dist, type_emb], dim=-1)

        # Pass through MLP
        pred = self.mlp(out)

        return pred
