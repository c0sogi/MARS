import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from library.config import Config


class HybridGNN(nn.Module):
    """
    Hybrid GNN-BiLSTM architecture for RNA degradation prediction.

    Combines a Graph Convolutional Network (GCN) to capture spatial folding structure
    with a Bidirectional LSTM to model sequential dependencies along the RNA backbone.
    """

    def __init__(self):
        super(HybridGNN, self).__init__()

        # =====================================================================
        # 1. Spatial Encoder (Graph Neural Network)
        # =====================================================================
        self.gnn_layers = nn.ModuleList()

        # First GNN Layer: Input Features -> Hidden Dim
        self.gnn_layers.append(GCNConv(Config.NUM_NODE_FEATURES, Config.GNN_HIDDEN_DIM))

        # Subsequent GNN Layers: Hidden Dim -> Hidden Dim
        # We subtract 1 because the first layer is already created
        for _ in range(Config.NUM_GNN_LAYERS - 1):
            self.gnn_layers.append(
                GCNConv(Config.GNN_HIDDEN_DIM, Config.GNN_HIDDEN_DIM)
            )

        self.gnn_dropout = nn.Dropout(Config.GNN_DROPOUT)

        # =====================================================================
        # 2. Sequential Refinement (BiLSTM)
        # =====================================================================
        # The GNN output becomes the input to the LSTM
        self.lstm = nn.LSTM(
            input_size=Config.GNN_HIDDEN_DIM,
            hidden_size=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.NUM_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.NUM_LSTM_LAYERS > 1 else 0.0,
        )

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        # Projects the LSTM hidden states to the target values
        # Input dimension is LSTM_HIDDEN_DIM * 2 because of bidirectionality
        self.fc_out = nn.Linear(Config.LSTM_HIDDEN_DIM * 2, Config.NUM_TARGETS)

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (torch_geometric.data.Batch): A batch of graph data containing:
                - x: Node features of shape (Total_Nodes, Num_Features)
                - edge_index: Graph connectivity of shape (2, Total_Edges)
                - batch: Batch vector mapping each node to a graph index (Total_Nodes,)

        Returns:
            torch.Tensor: Predictions of shape (Batch_Size, Seq_Length, Num_Targets)
        """
        x, edge_index = data.x, data.edge_index

        # --- 1. GNN Phase ---
        # Propagate information through the graph structure (backbone + pairs)
        for conv in self.gnn_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.gnn_dropout(x)

        # --- 2. Reshape Phase ---
        # PyTorch Geometric stacks all nodes from the batch into one dimension.
        # We need to reshape this to (Batch, Seq, Dim) for the LSTM.
        # We rely on the fixed sequence length of the RNA molecules.

        total_nodes = x.size(0)
        seq_len = Config.SEQ_LENGTH

        if total_nodes % seq_len != 0:
            raise ValueError(
                f"Total nodes ({total_nodes}) is not divisible by sequence length ({seq_len}). "
                "Ensure all graphs in the batch have the expected size."
            )

        batch_size = total_nodes // seq_len

        # Reshape: (Batch * Seq_Len, Hidden) -> (Batch, Seq_Len, Hidden)
        x = x.view(batch_size, seq_len, -1)

        # --- 3. LSTM Phase ---
        # Process the sequence with BiLSTM
        # lstm_out shape: (Batch, Seq_Len, LSTM_Hidden * 2)
        lstm_out, _ = self.lstm(x)

        # --- 4. Output Phase ---
        # Project to target dimensions
        # out shape: (Batch, Seq_Len, Num_Targets)
        out = self.fc_out(lstm_out)

        return out
