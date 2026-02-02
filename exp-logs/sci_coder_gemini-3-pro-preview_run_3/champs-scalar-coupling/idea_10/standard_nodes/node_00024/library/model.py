import torch
import torch.nn as nn
from torch_scatter import scatter
from library.config import Config
from library.layers import EdgeEmbedding, InteractionBlock, SharedCouplingHead


class DirectionalMPNN(nn.Module):
    """
    Optimized Node-Centric GNN (SchNet-variant).
    Replaces heavy Directional MPNN to enable training on full dataset (Cite Lesson 00014).
    Uses shared heads for data efficiency (Cite Lesson 00022).
    """

    def __init__(self):
        super(DirectionalMPNN, self).__init__()
        self.config = Config()

        # Hyperparameters
        hidden_dim = self.config.HIDDEN_DIM
        num_layers = self.config.NUM_LAYERS
        num_rbf = self.config.NUM_RBF
        num_atom_types = self.config.NUM_ATOM_TYPES

        # 1. Initial Embeddings
        self.node_embedding = nn.Embedding(num_atom_types, hidden_dim)

        self.edge_embedding = EdgeEmbedding(
            num_rbf=num_rbf,
            hidden_dim=hidden_dim,
            cutoff_lower=0.0,
            cutoff_upper=self.config.CUTOFF_RADIUS,
        )

        # 2. Message Passing Blocks (Node-Centric)
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(hidden_dim=hidden_dim, num_rbf=num_rbf)
                for _ in range(num_layers)
            ]
        )

        # 3. Readout
        self.readout = SharedCouplingHead(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            num_types=self.config.NUM_COUPLING_TYPES,
            hidden_dim=hidden_dim,
            output_dim=1,
        )

        # 4. Auxiliary Heads
        self.shielding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        # Unpack
        node_x = batch["node_x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]

        coupling_edge_index = batch["coupling_edge_index"]
        coupling_type = batch["coupling_type"]

        # 1. Embeddings
        h = self.node_embedding(node_x)

        if edge_index.shape[1] > 0:
            e = self.edge_embedding(edge_attr)

            # 2. Message Passing (Node Updates)
            for block in self.blocks:
                h = block(h, edge_index, e)
        else:
            e = torch.zeros(0, self.config.HIDDEN_DIM, device=h.device)

        # 3. Predictions
        pred_shielding = self.shielding_head(h)
        pred_charge = self.charge_head(h)

        if coupling_edge_index.shape[0] > 0:
            pred_coupling = self.readout(
                node_embeddings=h,
                edge_embeddings=e,
                edge_index=edge_index,
                coupling_edge_index=coupling_edge_index,
                coupling_type=coupling_type,
            )
        else:
            pred_coupling = torch.zeros(0, 1, device=h.device)

        return {
            "coupling": pred_coupling,
            "shielding": pred_shielding,
            "charge": pred_charge,
        }
