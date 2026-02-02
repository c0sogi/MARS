import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import configuration and utils
from library.config import GASEConfig
from library.graph_utils import MoleculeGraphBuilder

# -----------------------------------------------------------------------------
# 1. Model Architecture
# -----------------------------------------------------------------------------


class AtomEncoder(nn.Module):
    """
    Encodes initial node features (Atom Type + Bag of Neighbors) into hidden dimension.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeEncoder(nn.Module):
    """
    Encodes edge features (RBF + Inverse Distances) into hidden dimension.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class MessagePassingLayer(nn.Module):
    """
    Performs message passing:
    1. Compute messages m_ij = MLP(h_i || h_j || e_ij)
    2. Aggregate messages m_i = Sum(m_ji)
    3. Update h_i = h_i + MLP(m_i)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message function: Takes h_i, h_j, e_ij -> message
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        """
        h: (N_nodes, hidden_dim)
        edge_index: (2, N_edges)
        edge_attr: (N_edges, hidden_dim)
        """
        src, dst = edge_index

        # 1. Construct Message Inputs
        # Gather source and destination node features
        h_src = h[src]
        h_dst = h[dst]

        # Concatenate
        msg_input = torch.cat([h_src, h_dst, edge_attr], dim=1)

        # Compute Messages
        messages = self.message_mlp(msg_input)

        # 2. Aggregate (Scatter Sum)
        # We sum messages flowing INTO dst
        # Initialize buffer for aggregated messages
        aggr_messages = torch.zeros_like(h)
        # index_add_ is deterministic and reasonably fast
        aggr_messages.index_add_(0, dst, messages)

        # 3. Update
        update = self.update_mlp(aggr_messages)

        # Residual connection + LayerNorm
        h_new = self.norm(h + update)

        return h_new


class InteractionMPNN(nn.Module):
    """
    Main MPNN Model.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        # Node input: OneHot (5) + BagOfNeighbors (5) = 10
        self.node_input_dim = GASEConfig.NUM_ATOM_TYPES * 2
        # Edge input: RBF (16) + Inv (3) = 19
        self.edge_input_dim = GASEConfig.MPNN_NUM_RBF + 3

        self.hidden_dim = GASEConfig.MPNN_HIDDEN_DIM
        self.num_layers = GASEConfig.MPNN_NUM_LAYERS

        # Encoders
        self.atom_encoder = AtomEncoder(self.node_input_dim, self.hidden_dim)
        self.edge_encoder = EdgeEncoder(self.edge_input_dim, self.hidden_dim)

        # Message Passing Layers
        self.layers = nn.ModuleList(
            [MessagePassingLayer(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # Prediction Head (for auxiliary training)
        # Input to head is h_i || h_j || e_ij_target
        # Output is scalar coupling constant
        self.head_input_dim = self.hidden_dim * 2 + self.edge_input_dim
        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, pairs_idx, pairs_edge_attr):
        """
        Args:
            x: Node features (N, node_input_dim)
            edge_index: Graph connectivity (2, E)
            edge_attr: Graph edge features (E, edge_input_dim)
            pairs_idx: Indices of atom pairs to predict (P, 2) -> (idx_0, idx_1)
            pairs_edge_attr: Edge features for the specific pairs being predicted (P, edge_input_dim)
                             (Calculated on the fly or passed in, distinct from graph edges)

        Returns:
            pred: (P, 1) Scalar coupling constants
            embedding: (P, head_input_dim) The interaction embedding vector
        """
        # Encode
        h = self.atom_encoder(x)
        e = self.edge_encoder(edge_attr)

        # Propagate
        for layer in self.layers:
            h = layer(h, edge_index, e)

        # Readout for pairs
        idx_0, idx_1 = pairs_idx[:, 0], pairs_idx[:, 1]

        h_0 = h[idx_0]
        h_1 = h[idx_1]

        # Concatenate for Interaction Embedding
        # Note: We use the raw pair edge attributes here, not the encoded graph edge attributes
        # because the pair might not exist in the message passing graph, or we want the exact geometric info.
        embedding = torch.cat([h_0, h_1, pairs_edge_attr], dim=1)

        # Predict
        pred = self.head(embedding)

        return pred, embedding


# -----------------------------------------------------------------------------
# 2. Data Handling
# -----------------------------------------------------------------------------


class MoleculeDataset(Dataset):
    """
    Dataset that serves batches of molecules.
    Each item is a tuple: (node_feats, edge_indices, edge_feats, target_pairs, target_values, target_edge_feats)
    """

    def __init__(self, split, graph_data, metadata_df):
        self.split = split
        self.nodes = torch.from_numpy(graph_data["nodes"]).float()
        self.edge_indices = torch.from_numpy(graph_data["edge_indices"]).long()
        self.edge_features = torch.from_numpy(graph_data["edge_features"]).float()

        # Index dataframe: molecule_name -> slices
        self.index_df = graph_data["index"].set_index("molecule_name")

        # Metadata: Contains the targets (pairs)
        # Group metadata by molecule_name for fast retrieval
        # We only care about molecules that exist in both index and metadata
        valid_mols = self.index_df.index.intersection(
            metadata_df["molecule_name"].unique()
        )
        self.molecule_names = valid_mols.tolist()

        print(f"[{split}] Grouping metadata by molecule...")
        # Optimization: Sort metadata by molecule to speed up grouping
        meta_sorted = metadata_df[
            metadata_df["molecule_name"].isin(valid_mols)
        ].sort_values("molecule_name")

        # Create a dictionary of arrays for fast access
        # Key: molecule_name, Value: (indices, targets, distances)
        self.mol_data = {}

        for mol_name, group in meta_sorted.groupby("molecule_name"):
            indices = group[["atom_index_0", "atom_index_1"]].values.astype(np.int64)

            if "scalar_coupling_constant" in group.columns:
                targets = group["scalar_coupling_constant"].values.astype(np.float32)
            else:
                targets = np.zeros(len(group), dtype=np.float32)  # Test set

            if "dist" in group.columns:
                dists = group["dist"].values.astype(np.float32)
            else:
                # Fallback if dist missing (should not happen if pipeline followed)
                dists = np.zeros(len(group), dtype=np.float32)

            self.mol_data[mol_name] = (indices, targets, dists)

        # Pre-compute RBF constants for pair features
        self.rbf_min = GASEConfig.MPNN_RBF_MIN
        self.rbf_max = GASEConfig.MPNN_RBF_MAX
        self.num_rbf = GASEConfig.MPNN_NUM_RBF
        self.gamma = 1.0 / ((self.rbf_max - self.rbf_min) / self.num_rbf) ** 2
        self.centers = torch.linspace(self.rbf_min, self.rbf_max, self.num_rbf)

    def __len__(self):
        return len(self.molecule_names)

    def _compute_pair_edge_attr(self, dists):
        """Compute RBF + Inv features for target pairs."""
        dists = torch.from_numpy(dists).float().unsqueeze(1)  # (P, 1)

        # RBF
        rbf = torch.exp(-self.gamma * (dists - self.centers.unsqueeze(0)) ** 2)

        # Inv
        epsilon = 1e-6
        inv1 = 1.0 / (dists + epsilon)
        inv2 = 1.0 / (dists**2 + epsilon)
        inv3 = 1.0 / (dists**3 + epsilon)

        return torch.cat([rbf, inv1, inv2, inv3], dim=1)

    def __getitem__(self, idx):
        mol_name = self.molecule_names[idx]

        # 1. Get Graph Data
        idx_info = self.index_df.loc[mol_name]
        n_start = int(idx_info["node_start"])
        n_count = int(idx_info["node_count"])
        e_start = int(idx_info["edge_start"])
        e_count = int(idx_info["edge_count"])

        node_x = self.nodes[n_start : n_start + n_count]

        # Edge indices need to be shifted to be 0-based relative to this molecule
        # The stored indices are global indices. We subtract n_start from them.
        edge_idx = self.edge_indices[e_start : e_start + e_count] - n_start
        edge_attr = self.edge_features[e_start : e_start + e_count]

        # 2. Get Target Data
        pair_indices, targets, pair_dists = self.mol_data[mol_name]

        # Pair indices are local atom indices (0..28), so they match the 0-based node_x
        pair_indices = torch.from_numpy(pair_indices).long()
        targets = torch.from_numpy(targets).float().unsqueeze(1)

        pair_edge_attr = self._compute_pair_edge_attr(pair_dists)

        return node_x, edge_idx.T, edge_attr, pair_indices, targets, pair_edge_attr


def collate_mpnn(batch):
    """
    Collates a list of molecule data into a single batch graph.
    """
    # Unpack
    (
        node_feats_list,
        edge_index_list,
        edge_attr_list,
        pair_idx_list,
        target_list,
        pair_attr_list,
    ) = zip(*batch)

    # 1. Concatenate Nodes
    batch_x = torch.cat(node_feats_list, dim=0)

    # 2. Concatenate Edges with Index Shifting
    # We need to know the number of nodes in each graph to shift indices
    num_nodes_per_graph = [x.shape[0] for x in node_feats_list]
    cum_nodes = torch.cumsum(torch.tensor([0] + num_nodes_per_graph[:-1]), dim=0)

    shifted_edge_indices = []
    for i, edge_idx in enumerate(edge_index_list):
        shifted_edge_indices.append(edge_idx + cum_nodes[i])

    batch_edge_index = torch.cat(shifted_edge_indices, dim=1)
    batch_edge_attr = torch.cat(edge_attr_list, dim=0)

    # 3. Concatenate Pairs with Index Shifting
    shifted_pair_indices = []
    for i, pair_idx in enumerate(pair_idx_list):
        shifted_pair_indices.append(pair_idx + cum_nodes[i])

    batch_pair_idx = torch.cat(shifted_pair_indices, dim=0)
    batch_targets = torch.cat(target_list, dim=0)
    batch_pair_attr = torch.cat(pair_attr_list, dim=0)

    return (
        batch_x,
        batch_edge_index,
        batch_edge_attr,
        batch_pair_idx,
        batch_targets,
        batch_pair_attr,
    )


# -----------------------------------------------------------------------------
# 3. Training & Inference Logic
# -----------------------------------------------------------------------------


def run_mpnn_training(load_cached_data=True):
    """
    Main function to train the MPNN.
    """
    device = torch.device(GASEConfig.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Initializing MoleculeGraphBuilder...")
    builder = MoleculeGraphBuilder()

    # Load Graph Data
    train_graph = builder.process_data("train", load_cached_data=load_cached_data)
    val_graph = builder.process_data("val", load_cached_data=load_cached_data)

    # Load Metadata (Targets + Distances)
    from library.data_utils import process_and_cache_data

    df_train, df_val, _ = process_and_cache_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = MoleculeDataset("train", train_graph, df_train)
    val_dataset = MoleculeDataset("val", val_graph, df_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=GASEConfig.MPNN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_mpnn,
        num_workers=GASEConfig.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=GASEConfig.MPNN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_mpnn,
        num_workers=GASEConfig.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Setup
    model = InteractionMPNN().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=GASEConfig.MPNN_LEARNING_RATE,
        weight_decay=GASEConfig.MPNN_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=GASEConfig.MPNN_EPOCHS)
    criterion = nn.L1Loss()  # MAE

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting MPNN training...")
    for epoch in range(GASEConfig.MPNN_EPOCHS):
        model.train()

        for batch in train_loader:
            x, edge_index, edge_attr, pair_idx, targets, pair_attr = [
                b.to(device) for b in batch
            ]

            optimizer.zero_grad()
            preds, _ = model(x, edge_index, edge_attr, pair_idx, pair_attr)

            loss = criterion(preds, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        total_val_pairs = 0

        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, edge_attr, pair_idx, targets, pair_attr = [
                    b.to(device) for b in batch
                ]
                preds, _ = model(x, edge_index, edge_attr, pair_idx, pair_attr)
                loss = criterion(preds, targets)
                val_loss += loss.item() * targets.size(0)
                total_val_pairs += targets.size(0)

        val_loss /= total_val_pairs
        scheduler.step()

        print(f"Epoch {epoch+1}/{GASEConfig.MPNN_EPOCHS} | Val MAE: {val_loss:.8f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), GASEConfig.MPNN_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= GASEConfig.MPNN_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MAE: {best_val_loss:.8f}")

    # Clean up
    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()


def generate_embeddings(load_cached_data=True):
    """
    Generates embeddings for Train, Val, and Test sets using the trained MPNN.
    Saves them to .npy files specified in Config.
    """
    device = torch.device(GASEConfig.DEVICE)

    # Load Model
    if not os.path.exists(GASEConfig.MPNN_MODEL_PATH):
        raise FileNotFoundError("MPNN model not found. Run training first.")

    model = InteractionMPNN().to(device)
    model.load_state_dict(torch.load(GASEConfig.MPNN_MODEL_PATH, map_location=device))
    model.eval()

    # Data Setup
    builder = MoleculeGraphBuilder()
    from library.data_utils import process_and_cache_data

    df_train, df_val, df_test = process_and_cache_data(
        load_cached_data=load_cached_data
    )

    splits = [
        ("train", df_train, GASEConfig.EMBEDDINGS_TRAIN_PATH),
        ("val", df_val, GASEConfig.EMBEDDINGS_VAL_PATH),
        ("test", df_test, GASEConfig.EMBEDDINGS_TEST_PATH),
    ]

    for split_name, df_meta, save_path in splits:
        print(f"Generating embeddings for {split_name}...")

        graph_data = builder.process_data(split_name, load_cached_data=load_cached_data)
        dataset = MoleculeDataset(split_name, graph_data, df_meta)

        loader = DataLoader(
            dataset,
            batch_size=GASEConfig.MPNN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_mpnn,
            num_workers=GASEConfig.NUM_WORKERS,
            pin_memory=True,
        )

        embeddings_list = []

        with torch.no_grad():
            for batch in loader:
                x, edge_index, edge_attr, pair_idx, _, pair_attr = [
                    b.to(device) for b in batch
                ]

                # We only need embeddings
                _, embeddings = model(x, edge_index, edge_attr, pair_idx, pair_attr)
                embeddings_list.append(embeddings.cpu().numpy())

        # Concatenate
        full_embeddings = np.concatenate(embeddings_list, axis=0)

        # Save
        np.save(save_path, full_embeddings)
        print(f"Saved embeddings to {save_path}. Shape: {full_embeddings.shape}")

        # Cleanup
        del dataset, loader, graph_data, embeddings_list, full_embeddings
        gc.collect()
