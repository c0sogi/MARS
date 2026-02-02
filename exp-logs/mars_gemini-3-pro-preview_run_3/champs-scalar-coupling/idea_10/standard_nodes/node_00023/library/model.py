import torch
import torch.nn as nn
from torch_scatter import scatter
from library.config import Config
from library.layers import EdgeEmbedding, InteractionBlock, SharedCouplingHead


class DirectionalMPNN(nn.Module):
    """
    Directional Message Passing Neural Network (DMPNN) for Scalar Coupling Prediction.

    This model operates on directed edges (bonds) to explicitly model bond angles
    via triplet interactions. It updates edge embeddings through several message
    passing layers and then aggregates information back to nodes for auxiliary
    predictions and the final edge-conditioned readout.
    """

    def __init__(self):
        super(DirectionalMPNN, self).__init__()
        self.config = Config()

        # Hyperparameters
        hidden_dim = self.config.HIDDEN_DIM
        num_layers = self.config.NUM_LAYERS
        num_rbf = self.config.NUM_RBF
        num_angle_rbf = self.config.NUM_ANGLE_RBF
        num_atom_types = self.config.NUM_ATOM_TYPES

        # 1. Initial Embeddings
        # Node embedding: Maps atom type index to hidden vector
        self.node_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # Edge embedding: Maps scalar distance to hidden vector via RBF + MLP
        self.edge_embedding = EdgeEmbedding(
            num_rbf=num_rbf,
            hidden_dim=hidden_dim,
            cutoff_lower=0.0,
            cutoff_upper=self.config.CUTOFF_RADIUS,
        )

        # 2. Message Passing Blocks
        # Stack of InteractionBlocks that update edge features based on angular context
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(hidden_dim=hidden_dim, num_angle_rbf=num_angle_rbf)
                for _ in range(num_layers)
            ]
        )

        # 3. Node Update / Aggregation
        # After edge updates, we aggregate incoming edges to update node features
        # This is crucial for the auxiliary tasks and the node-component of the readout
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4. Primary Readout (Coupling Constant)
        # Predicts coupling constant using updated node features and specific edge features
        # Cite Lesson 00022: Using SharedCouplingHead instead of TypeSpecificHeads
        self.readout = SharedCouplingHead(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            num_types=self.config.NUM_COUPLING_TYPES,
            type_emb_dim=self.config.TYPE_EMB_DIM,
            hidden_dim=hidden_dim,
            output_dim=1,
        )

        # 5. Auxiliary Heads
        # Predict Magnetic Shielding and Mulliken Charges from updated node features
        # These act as regularizers during training
        self.shielding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (dict): Dictionary containing collated graph data tensors.
                          Keys: node_x, edge_index, edge_attr, triplet_index, triplet_attr, etc.

        Returns:
            dict: Dictionary containing predictions:
                  - 'coupling': (num_couplings, 1)
                  - 'shielding': (num_nodes, 1)
                  - 'charge': (num_nodes, 1)
        """
        # Unpack necessary data
        node_x = batch["node_x"]  # (N,)
        edge_index = batch["edge_index"]  # (2, M)
        edge_attr = batch["edge_attr"]  # (M,)
        triplet_index = batch["triplet_index"]  # (2, K)
        triplet_attr = batch["triplet_attr"]  # (K,)

        # Coupling specific indices
        coupling_edge_index = batch["coupling_edge_index"]  # (C,)
        coupling_type = batch["coupling_type"]  # (C,)

        # 1. Initialize Embeddings
        h = self.node_embedding(node_x)  # (N, hidden_dim)

        # Handle case with no edges (e.g., single atom molecules if any, though unlikely in this dataset)
        if edge_index.shape[1] > 0:
            e = self.edge_embedding(edge_attr)  # (M, hidden_dim)

            # 2. Directional Message Passing
            # Iteratively update edge embeddings
            for block in self.blocks:
                if triplet_index.shape[1] > 0:
                    e = block(e, triplet_index, triplet_attr)
                else:
                    # If no triplets (no angles), skip interaction or just pass through
                    # In this architecture, InteractionBlock is residual, so e remains e
                    pass

            # 3. Aggregate Messages to Nodes
            # Sum incoming edge features to update node features
            # edge_index[1] is the target node index for directed edges
            # e contains info about the edge and its incoming neighbors
            dst_idx = edge_index[1]
            aggregated_messages = scatter(
                e, dst_idx, dim=0, dim_size=h.size(0), reduce="sum"
            )

            # Update node features with aggregated messages (Residual connection)
            h = h + self.node_update_mlp(aggregated_messages)
        else:
            # If no edges, create dummy edge embeddings for readout consistency if needed
            # (Though coupling_edge_index would be empty, so readout won't be called on them)
            e = torch.zeros(0, self.config.HIDDEN_DIM, device=h.device)

        # 4. Generate Predictions

        # A. Auxiliary Tasks (Node-level)
        pred_shielding = self.shielding_head(h)
        pred_charge = self.charge_head(h)

        # B. Primary Task (Edge-level / Pair-level)
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
