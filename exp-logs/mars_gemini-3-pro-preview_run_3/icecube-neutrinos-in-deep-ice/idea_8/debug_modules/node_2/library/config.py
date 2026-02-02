import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")

    # Data Parameters
    MAX_PULSES = 196
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    # Model Parameters
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    K_NEIGHBORS = 20
    DROPOUT = 0.1

    # Training Parameters
    EPOCHS = 15  # Adjusted for 24h limit
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        # Set seeds
        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.SEED)


# -----------------------------------------------------------------------------
# 2. Data Processing & Caching
# -----------------------------------------------------------------------------
def load_geometry():
    """Loads sensor geometry and returns a dict mapping sensor_id to [x, y, z]."""
    df = pd.read_csv(Config.SENSOR_GEO_PATH)
    # Assuming sensor_id corresponds to the index if not explicitly present,
    # but usually there is a sensor_id column.
    if "sensor_id" in df.columns:
        sensor_ids = df["sensor_id"].values
    else:
        sensor_ids = df.index.values

    coords = df[["x", "y", "z"]].values.astype(np.float32)
    return dict(zip(sensor_ids, coords))


def get_canonical_transform(xyz, charge, time):
    """
    Computes the rotation matrix to align the event's principal axis with Z.
    Returns the transformed coordinates.
    """
    # 1. Center of Gravity
    weights = charge / (charge.sum() + 1e-6)
    cog = np.sum(xyz * weights[:, None], axis=0)
    xyz_centered = xyz - cog

    # 2. SVD on weighted covariance
    # Covariance: (X - mu).T @ W @ (X - mu)
    # We can just weight the centered coordinates
    weighted_xyz = xyz_centered * np.sqrt(weights[:, None])
    cov = weighted_xyz.T @ weighted_xyz

    try:
        U, S, Vh = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return xyz_centered  # Fallback

    # Principal axis is the first column of U (or row of Vh)
    axis = U[:, 0]

    # 3. Orientation Correction (align with time)
    # Project positions onto axis
    projections = xyz_centered @ axis
    # Correlation with time
    if np.std(projections) > 1e-6 and np.std(time) > 1e-6:
        corr = np.corrcoef(projections, time)[0, 1]
        if corr < 0:
            axis = -axis
            U[:, 0] = -U[:, 0]

    # 4. Rotate
    # We want to map the principal axis (U[:, 0]) to Z (0, 0, 1)
    # Actually, projecting onto U aligns the cloud with the canonical axes (X, Y, Z)
    # defined by the eigenvectors.
    xyz_transformed = xyz_centered @ U

    return xyz_transformed


def process_batch(batch_id, meta_df, sensor_map, mode="train", load_cached_data=True):
    """
    Processes a single batch of events.
    mode: 'train' (returns X, y) or 'test' (returns X, ids)
    """
    cache_file_base = os.path.join(Config.CACHE_DIR, f"{mode}_batch_{batch_id}")
    file_X_raw = cache_file_base + "_X_raw.npy"
    file_X_canon = cache_file_base + "_X_canon.npy"
    file_meta = cache_file_base + "_meta.npy"  # Stores y or ids depending on mode

    # 1. Try Load Cache
    if load_cached_data:
        if (
            os.path.exists(file_X_raw)
            and os.path.exists(file_X_canon)
            and os.path.exists(file_meta)
        ):
            X_raw = np.load(file_X_raw)
            X_canon = np.load(file_X_canon)
            meta = np.load(file_meta)
            return X_raw, X_canon, meta

    # 2. Process from Scratch
    # Load raw parquet
    if mode == "train" or mode == "val":
        file_path = os.path.join(Config.INPUT_DIR, "train", f"batch_{batch_id}.parquet")
    else:
        file_path = os.path.join(Config.INPUT_DIR, "test", f"batch_{batch_id}.parquet")

    batch_df = pd.read_parquet(file_path)

    # Filter metadata for this batch
    batch_meta = meta_df[meta_df["batch_id"] == batch_id].copy()

    # Prepare containers
    n_events = len(batch_meta)
    X_raw = np.zeros(
        (n_events, Config.MAX_PULSES, 6), dtype=np.float32
    )  # x,y,z,t,q,aux
    X_canon = np.zeros((n_events, Config.MAX_PULSES, 6), dtype=np.float32)

    if mode in ["train", "val"]:
        targets = np.zeros((n_events, 2), dtype=np.float32)  # azimuth, zenith
    else:
        ids = np.zeros((n_events,), dtype=np.int64)

    # Group by event_id
    # To speed up, we can use the index if it's sorted, but groupby is safer
    events_group = batch_df.groupby("event_id")

    # Iterate
    # We need to map event_id to index in our arrays
    event_id_to_idx = {eid: i for i, eid in enumerate(batch_meta["event_id"].values)}

    for eid, group in events_group:
        if eid not in event_id_to_idx:
            continue
        idx = event_id_to_idx[eid]

        # Get features
        sensor_ids = group["sensor_id"].values
        time = group["time"].values.astype(np.float32)
        charge = group["charge"].values.astype(np.float32)
        aux = group["auxiliary"].values.astype(np.float32)

        # Map geometry
        # Vectorized map lookup
        # Note: sensor_map is a dict. For speed, could convert to array if IDs are contiguous,
        # but they are 5160, so simple lookup is okay or list comprehension.
        xyz = np.array([sensor_map[sid] for sid in sensor_ids], dtype=np.float32)

        # Sampling
        n_pulses = len(time)
        if n_pulses > Config.MAX_PULSES:
            # Hybrid Sampling: 50% high charge, 50% early time
            n_high_q = Config.MAX_PULSES // 2
            n_early_t = Config.MAX_PULSES - n_high_q

            # Indices for high charge
            idx_q = np.argsort(charge)[-n_high_q:]

            # Remaining indices for time
            mask = np.ones(n_pulses, dtype=bool)
            mask[idx_q] = False
            remaining_indices = np.where(mask)[0]

            if len(remaining_indices) > n_early_t:
                idx_t_sub = np.argsort(time[remaining_indices])[:n_early_t]
                idx_t = remaining_indices[idx_t_sub]
            else:
                idx_t = remaining_indices

            indices = np.concatenate([idx_q, idx_t])
            indices = np.sort(indices)  # Sort by original index (usually time-ish)
        else:
            indices = np.arange(n_pulses)

        # Select Data
        s_xyz = xyz[indices]
        s_time = time[indices]
        s_charge = charge[indices]
        s_aux = aux[indices]

        # Normalize time (relative to start) and charge (log)
        s_time_norm = (s_time - s_time.min()) / 1000.0  # scale to us roughly
        s_charge_norm = np.log10(s_charge + 1.0)

        # Pad if necessary
        n_selected = len(indices)

        # Raw Features
        X_raw[idx, :n_selected, 0:3] = s_xyz
        X_raw[idx, :n_selected, 3] = s_time_norm
        X_raw[idx, :n_selected, 4] = s_charge_norm
        X_raw[idx, :n_selected, 5] = s_aux

        # Canonical Transform
        try:
            xyz_canon = get_canonical_transform(s_xyz, s_charge, s_time)
        except:
            xyz_canon = s_xyz  # Fallback

        X_canon[idx, :n_selected, 0:3] = xyz_canon
        X_canon[idx, :n_selected, 3] = s_time_norm
        X_canon[idx, :n_selected, 4] = s_charge_norm
        X_canon[idx, :n_selected, 5] = s_aux

        # Targets/IDs
        if mode in ["train", "val"]:
            row = batch_meta.iloc[idx]
            targets[idx, 0] = row["azimuth"]
            targets[idx, 1] = row["zenith"]
        else:
            ids[idx] = eid

    # 3. Save Cache
    np.save(file_X_raw, X_raw)
    np.save(file_X_canon, X_canon)
    if mode in ["train", "val"]:
        np.save(file_meta, targets)
        return X_raw, X_canon, targets
    else:
        np.save(file_meta, ids)
        return X_raw, X_canon, ids


# -----------------------------------------------------------------------------
# 3. Dataset
# -----------------------------------------------------------------------------
class IceCubeDataset(Dataset):
    def __init__(self, X_raw, X_canon, y=None):
        self.X_raw = torch.tensor(X_raw, dtype=torch.float32)
        self.X_canon = torch.tensor(X_canon, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_raw[idx], self.X_canon[idx], self.y[idx]
        else:
            return self.X_raw[idx], self.X_canon[idx]


# -----------------------------------------------------------------------------
# 4. Model: Dual-View Attentive Graph Network
# -----------------------------------------------------------------------------
class DynEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, batch_idx=None):
        # x: [B, N, C]
        B, N, C = x.shape

        # k-NN graph construction based on spatial coordinates (first 3 channels)
        # We flatten batch to use cdist efficiently or loop?
        # For B=128, N=196, loop is fast enough on GPU.

        x_out = []
        for i in range(B):
            xi = x[i]  # [N, C]
            pos = xi[:, :3]  # Use x,y,z for distance

            # Simple KNN
            dist = torch.cdist(pos, pos)  # [N, N]
            # Mask self loop? No, usually included or excluded. Let's exclude self
            # by adding infinity to diagonal
            dist.diagonal().fill_(float("inf"))

            _, idx = dist.topk(self.k, largest=False)  # [N, k]

            # Gather neighbors
            # neighbors: [N, k, C]
            neighbors = xi[idx]

            # Edge features: [xi, xj - xi]
            xi_expanded = xi.unsqueeze(1).expand(-1, self.k, -1)  # [N, k, C]
            edge_feat = torch.cat(
                [xi_expanded, neighbors - xi_expanded], dim=-1
            )  # [N, k, 2C]

            # MLP
            out = self.mlp(edge_feat)  # [N, k, out_channels]

            # Aggregation (Max)
            out = out.max(dim=1)[0]  # [N, out_channels]
            x_out.append(out)

        return torch.stack(x_out)  # [B, N, out_channels]


class DV_AGN(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = 6  # x, y, z, t, q, aux

        # Embeddings
        self.embed_raw = nn.Linear(input_dim, Config.EMBED_DIM)
        self.embed_canon = nn.Linear(input_dim, Config.EMBED_DIM)

        # Encoders
        self.gnn_raw = nn.ModuleList(
            [
                DynEdgeConv(Config.EMBED_DIM, Config.EMBED_DIM, k=Config.K_NEIGHBORS),
                DynEdgeConv(Config.EMBED_DIM, Config.EMBED_DIM, k=Config.K_NEIGHBORS),
            ]
        )

        self.gnn_canon = nn.ModuleList(
            [
                DynEdgeConv(Config.EMBED_DIM, Config.EMBED_DIM, k=Config.K_NEIGHBORS),
                DynEdgeConv(Config.EMBED_DIM, Config.EMBED_DIM, k=Config.K_NEIGHBORS),
            ]
        )

        # Cross Attention Fusion
        # Query: Canonical, Key/Value: Raw
        self.query = nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM)
        self.key = nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM)
        self.value = nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM)
        self.scale = math.sqrt(Config.EMBED_DIM)

        # Global Pooling Attention
        self.pool_attn = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, 1),
            nn.Tanh(),  # Tanh often used for attention scores
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 3),  # nx, ny, nz
        )

    def forward(self, x_raw, x_canon):
        # x: [B, N, 6]

        # Embedding
        h_raw = self.embed_raw(x_raw)
        h_canon = self.embed_canon(x_canon)

        # GNN Streams
        for layer in self.gnn_raw:
            h_raw = layer(h_raw) + h_raw  # Residual

        for layer in self.gnn_canon:
            h_canon = layer(h_canon) + h_canon  # Residual

        # Cross Attention Fusion
        # We want to enhance Canonical features with Raw context
        Q = self.query(h_canon)  # [B, N, D]
        K = self.key(h_raw)  # [B, N, D]
        V = self.value(h_raw)  # [B, N, D]

        # Attention scores: Q @ K.T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, N, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        h_fused = torch.matmul(attn_weights, V)  # [B, N, D]
        h_final = h_canon + h_fused  # Residual connection

        # Global Pooling
        # Weighted sum based on learned attention
        pool_weights = torch.softmax(self.pool_attn(h_final), dim=1)  # [B, N, 1]
        h_global = torch.sum(h_final * pool_weights, dim=1)  # [B, D]

        # Prediction
        out = self.head(h_global)
        return out


# -----------------------------------------------------------------------------
# 5. Training & Inference Utils
# -----------------------------------------------------------------------------
def direction_loss(pred, true_azimuth, true_zenith):
    # Convert true angles to vector
    sa = torch.sin(true_azimuth)
    ca = torch.cos(true_azimuth)
    sz = torch.sin(true_zenith)
    cz = torch.cos(true_zenith)

    true_x = ca * sz
    true_y = sa * sz
    true_z = cz

    true_vec = torch.stack([true_x, true_y, true_z], dim=1)

    # Normalize pred
    pred = F.normalize(pred, p=2, dim=1)

    # Cosine similarity
    cos_sim = torch.sum(pred * true_vec, dim=1)
    # Loss = 1 - cos_sim (minimize angle)
    return 1.0 - cos_sim.mean()


def train_model(load_cached_data=True):
    Config.setup()
    sensor_map = load_geometry()

    # Load Metadata
    train_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "train_metadata.parquet")
    )
    val_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "val_metadata.parquet")
    )

    # Limit for debugging/time constraints if needed
    # train_meta = train_meta.iloc[:100000]

    # Get Batch IDs
    train_batches = train_meta["batch_id"].unique()
    val_batches = val_meta["batch_id"].unique()

    model = DV_AGN().to(Config.DEVICE)
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler placeholder (needs total steps)
    scheduler = None

    best_val_loss = float("inf")
    patience = 3
    no_improve = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0
        train_batches_count = 0

        # Shuffle batches
        np.random.shuffle(train_batches)

        for batch_id in train_batches:
            # Load/Process Batch
            X_raw, X_canon, y = process_batch(
                batch_id, train_meta, sensor_map, "train", load_cached_data
            )

            dataset = IceCubeDataset(X_raw, X_canon, y)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            if scheduler is None:
                total_steps = len(train_batches) * len(loader) * Config.EPOCHS
                scheduler = OneCycleLR(
                    optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps
                )

            for xr, xc, targets in loader:
                xr, xc = xr.to(Config.DEVICE), xc.to(Config.DEVICE)
                az, ze = targets[:, 0].to(Config.DEVICE), targets[:, 1].to(
                    Config.DEVICE
                )

                optimizer.zero_grad()
                pred = model(xr, xc)
                loss = direction_loss(pred, az, ze)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss_sum += loss.item()

            train_batches_count += 1
            # Clean up
            del X_raw, X_canon, y, dataset, loader
            gc.collect()

        avg_train_loss = train_loss_sum / train_batches_count

        # Validation
        model.eval()
        val_errors = []

        with torch.no_grad():
            for batch_id in val_batches:
                X_raw, X_canon, y = process_batch(
                    batch_id, val_meta, sensor_map, "val", load_cached_data
                )
                dataset = IceCubeDataset(X_raw, X_canon, y)
                loader = DataLoader(
                    dataset, batch_size=Config.BATCH_SIZE, shuffle=False
                )

                for xr, xc, targets in loader:
                    xr, xc = xr.to(Config.DEVICE), xc.to(Config.DEVICE)
                    az, ze = targets[:, 0].numpy(), targets[:, 1].numpy()

                    pred = model(xr, xc)
                    errors = angular_error(pred, az, ze)
                    val_errors.extend(errors)

                del X_raw, X_canon, y, dataset, loader
                gc.collect()

        avg_val_error = np.mean(val_errors)
        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val MAE={avg_val_error:.6f}"
        )

        if avg_val_error < best_val_loss:
            best_val_loss = avg_val_error
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "model.pth")
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break


def generate_submission(load_cached_data=True):
    Config.setup()
    sensor_map = load_geometry()
    model = DV_AGN().to(Config.DEVICE)

    model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    if not os.path.exists(model_path):
        print("No trained model found. Skipping submission.")
        return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    test_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "test_metadata.parquet")
    )
    test_batches = test_meta["batch_id"].unique()

    all_ids = []
    all_azimuth = []
    all_zenith = []

    print("Generating submission...")
    with torch.no_grad():
        for batch_id in test_batches:
            X_raw, X_canon, ids = process_batch(
                batch_id, test_meta, sensor_map, "test", load_cached_data
            )
            dataset = IceCubeDataset(X_raw, X_canon)
            loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            for xr, xc in loader:
                xr, xc = xr.to(Config.DEVICE), xc.to(Config.DEVICE)
                pred = model(xr, xc)
                pred = F.normalize(pred, p=2, dim=1).cpu().numpy()

                # Convert vec to angles
                # z = cos(zenith) -> zenith = acos(z)
                # x = cos(az)sin(zenith), y = sin(az)sin(zenith) -> az = atan2(y, x)

                zenith = np.arccos(np.clip(pred[:, 2], -1, 1))
                azimuth = np.arctan2(pred[:, 1], pred[:, 0])
                # Azimuth in [0, 2pi]
                azimuth[azimuth < 0] += 2 * np.pi

                all_azimuth.extend(azimuth)
                all_zenith.extend(zenith)

            all_ids.extend(ids)
            del X_raw, X_canon, ids, dataset, loader
            gc.collect()

    # Save
    sub_df = pd.DataFrame(
        {"event_id": all_ids, "azimuth": all_azimuth, "zenith": all_zenith}
    )
    sub_df.to_csv(os.path.join(Config.SUBMISSION_DIR, "submission.csv"), index=False)
    print("Submission saved.")
