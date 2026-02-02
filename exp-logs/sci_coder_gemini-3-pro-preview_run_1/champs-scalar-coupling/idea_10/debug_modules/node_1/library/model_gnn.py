import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
import pandas as pd
import numpy as np
import random
from library import config, utils, graph_dataset


# Set fixed seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(config.RANDOM_STATE)


class InteractionMPNN(nn.Module):
    """
    Message Passing Neural Network for Scalar Coupling Prediction.
    Extracts learned interaction embeddings for downstream tasks.
    """

    def __init__(self):
        super(InteractionMPNN, self).__init__()

        # Hyperparameters from config
        node_dim = config.GNN_PARAMS["node_dim"]
        hidden_dim = config.GNN_PARAMS["hidden_dim"]
        num_layers = config.GNN_PARAMS["num_layers"]
        num_rbf = config.GNN_PARAMS["num_rbf"]
        cutoff = config.GNN_PARAMS["cutoff"]

        # 1. Embeddings
        # Atomic numbers 1, 6, 7, 8, 9 mapped to 0-4
        self.atom_emb = nn.Embedding(5, hidden_dim)

        # Edge Projection: RBF -> Hidden
        self.edge_proj = nn.Linear(num_rbf, hidden_dim)

        # Coupling Type Embedding (8 types)
        self.type_emb = nn.Embedding(8, hidden_dim)

        # RBF Expander (Re-instantiated here to be part of the model)
        self.rbf = graph_dataset.GaussianSmearing(
            start=0.0, stop=cutoff, num_gaussians=num_rbf
        )

        # 2. Message Passing Layers (GINEConv)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # MLP for GINEConv: h_i' = MLP( (1+eps)h_i + sum(ReLU(h_j + e_ij)) )
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(nn_layer, train_eps=True))

        # 3. Readout / Prediction Head
        # Input: h_i (hidden) + h_j (hidden) + e_ij (hidden) + type_emb (hidden)
        readout_input_dim = hidden_dim * 4

        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, pos = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.pos,
        )

        # Embed Atoms
        x = self.atom_emb(x.squeeze())

        # Embed Graph Edges (Pre-computed RBFs in data.edge_attr)
        edge_emb = self.edge_proj(edge_attr)

        # Message Passing
        for conv in self.convs:
            x = conv(x, edge_index, edge_emb)
            x = F.relu(x)

        # --- Interaction Extraction ---
        # Get indices of coupling pairs
        idx0, idx1 = data.coupling_edge_index

        # Gather Node Embeddings
        h0 = x[idx0]
        h1 = x[idx1]

        # Compute Pairwise Edge Embedding (on-the-fly)
        # We calculate distance between coupling atoms directly
        pos0 = pos[idx0]
        pos1 = pos[idx1]
        dist_vec = (pos0 - pos1).norm(dim=1)

        # Expand distance and project
        dist_rbf = self.rbf(dist_vec)
        pair_edge_emb = self.edge_proj(dist_rbf)

        # Get Type Embedding
        t_emb = self.type_emb(data.coupling_type_idx)

        # Concatenate for Readout
        # The "Learned Interaction Embedding" is [h0, h1, pair_edge_emb]
        interaction_emb = torch.cat([h0, h1, pair_edge_emb], dim=1)

        # Full input for prediction
        readout_in = torch.cat([interaction_emb, t_emb], dim=1)

        # Predict
        out = self.readout(readout_in)

        return out.squeeze(), interaction_emb


def train_gnn(load_cached_data=True):
    """
    Trains the GNN model and saves the best checkpoint.
    """
    device = torch.device(config.GNN_PARAMS["device"])
    print(f"Training GNN on {device}...")

    # Load Datasets
    train_dataset = graph_dataset.get_dataset("train", load_cached_data)
    val_dataset = graph_dataset.get_dataset("val", load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.GNN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.GNN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Initialize Model
    model = InteractionMPNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.GNN_PARAMS["learning_rate"],
        weight_decay=config.GNN_PARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = nn.L1Loss()

    # Training Loop
    best_val_mae = float("inf")
    early_stop_counter = 0
    patience = 7

    model_save_path = os.path.join(config.MODEL_DIR, "gnn_best.pt")

    for epoch in range(config.GNN_PARAMS["epochs"]):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred, _ = model(batch)
            loss = criterion(pred, batch.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        avg_train_loss = train_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs

        avg_val_loss = val_loss / len(val_dataset)

        # Scheduler Step
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{config.GNN_PARAMS['epochs']} | Train MAE: {avg_train_loss:.8f} | Val MAE: {avg_val_loss:.8f}"
        )

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_mae:
            best_val_mae = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
            print(f"  New best model saved to {model_save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MAE: {best_val_mae:.8f}")
    return model


def generate_embeddings(split, load_cached_data=True):
    """
    Generates or loads learned interaction embeddings for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and embedding columns.
    """
    # Cache path
    cache_file = os.path.join(config.CACHE_DIR, f"gnn_embeddings_{split}.parquet")

    # Check cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached embeddings for {split} from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Generating embeddings for {split}...")

    # Load Model
    device = torch.device(config.GNN_PARAMS["device"])
    model_path = os.path.join(config.MODEL_DIR, "gnn_best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first."
        )

    model = InteractionMPNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Dataset
    dataset = graph_dataset.get_dataset(split, load_cached_data=True)
    loader = DataLoader(
        dataset,
        batch_size=config.GNN_PARAMS["batch_size"] * 2,
        shuffle=False,
        num_workers=0,
    )

    ids_list = []
    embeddings_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, emb = model(batch)

            ids_list.append(batch.id.cpu().numpy())
            embeddings_list.append(emb.cpu().numpy())

    # Concatenate
    all_ids = np.concatenate(ids_list)
    all_embs = np.concatenate(embeddings_list)

    # Create DataFrame
    # Columns: id, emb_0, emb_1, ...
    cols = [f"gnn_emb_{i}" for i in range(all_embs.shape[1])]

    # Optimize memory: use float32
    df_emb = pd.DataFrame(all_embs, columns=cols, dtype=np.float32)
    df_emb.insert(0, "id", all_ids)

    # Save to cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    print(f"Saving embeddings to {cache_file}")
    df_emb.to_parquet(cache_file, index=False)

    return df_emb
