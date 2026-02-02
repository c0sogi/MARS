import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add

from library.config import Config
from library.data import MolecularGraphDataset, collate_graphs
from library.utils import TargetScaler


class DMPNNLayer(nn.Module):
    """
    Directional Message Passing Layer.
    Operates on directed edges (u->v) and aggregates messages from incoming edges (k->u)
    modulated by the bond angle (k-u-v) encoded in SBF.
    """

    def __init__(self, hidden_dim, num_rbf, num_sbf):
        super(DMPNNLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # Interaction Block: Generates messages from triplets
        self.lin_kj = nn.Linear(hidden_dim, hidden_dim)
        self.lin_sbf = nn.Linear(num_sbf * num_rbf, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)

        # Update Block: Updates edge embeddings
        self.lin_ji = nn.Linear(hidden_dim, hidden_dim)
        self.lin_rbf = nn.Linear(num_rbf, hidden_dim)

        # Deep MLP for non-linear update
        self.mlp_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, h, rbf, sbf, triplet_indices):
        """
        Args:
            h: (E, hidden_dim) Edge embeddings.
            rbf: (E, num_rbf) Radial basis features for edges.
            sbf: (T, num_sbf * num_rbf) Spherical basis features for triplets.
            triplet_indices: (2, T) Indices [incoming_edge_idx, outgoing_edge_idx].
        """
        # 1. Message Generation
        # Incoming edges (k->j)
        idx_kj = triplet_indices[0]
        # Outgoing edges (j->i)
        idx_ji = triplet_indices[1]

        # Transform incoming edge features
        h_kj = self.lin_kj(h)  # (E, dim)
        feat_kj = h_kj[idx_kj]  # (T, dim)

        # Transform angular features
        feat_sbf = self.lin_sbf(sbf)  # (T, dim)

        # Interaction (Hadamard product)
        msg_triplet = feat_kj * feat_sbf
        msg_triplet = self.lin_msg(msg_triplet)  # (T, dim)

        # Aggregate messages to outgoing edges
        # Sum messages for each outgoing edge index
        num_edges = h.size(0)
        m_ji = torch.zeros(num_edges, self.hidden_dim, device=h.device)
        # Using index_add_ for efficient aggregation
        m_ji.index_add_(0, idx_ji, msg_triplet)

        # 2. State Update
        h_ji_trans = self.lin_ji(h)
        rbf_trans = self.lin_rbf(rbf)

        # Concatenate: transformed self, aggregated message, geometric feature
        update_input = torch.cat([h_ji_trans, m_ji, rbf_trans], dim=1)

        # Residual update
        h_update = self.mlp_update(update_input)

        return h + h_update


class GlobalAttention(nn.Module):
    """
    Global Interaction Module using a Transformer Encoder.
    Allows every atom to attend to every other atom in the molecule.
    """

    def __init__(self, hidden_dim, n_heads, n_layers, d_ff, dropout=0.1):
        super(GlobalAttention, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, batch_atom):
        """
        Args:
            x: (N, hidden_dim) Atom embeddings.
            batch_atom: (N,) Batch indices for atoms.
        Returns:
            x_final: (N, hidden_dim) Updated atom embeddings.
        """
        # Convert to dense batch: (B, max_nodes, dim)
        x_dense, mask = to_dense_batch(x, batch_atom)

        # Create padding mask for Transformer (True where padded)
        # mask is True where real, False where padded.
        # Transformer expects src_key_padding_mask to be True for ignored positions.
        padding_mask = ~mask

        # Apply Transformer
        x_trans = self.transformer(x_dense, src_key_padding_mask=padding_mask)

        # Flatten back to (N, dim)
        x_final = x_trans[mask]

        return x_final


class HGANet(nn.Module):
    """
    Hybrid Geometric-Attention Network.
    Pipeline:
    1. Local Geometric Encoder (DMPNN) on directed edges.
    2. Global Aggregation (Edges -> Nodes).
    3. Global Transformer (Nodes -> Nodes).
    4. Deterministic Fusion Readout (Node u, Node v, Edge uv).
    """

    def __init__(self, config=Config):
        super(HGANet, self).__init__()
        self.hidden_dim = config.HIDDEN_DIM

        # --- Embeddings ---
        # Atomic numbers (H=1, C=6, N=7, O=8, F=9). Max index around 10.
        self.atom_embedding = nn.Embedding(20, self.hidden_dim)

        # Coupling Types (8 types)
        self.type_embedding = nn.Embedding(8, self.hidden_dim)

        # --- Initial Edge Embedding ---
        # Input: Atom_u, Atom_v, RBF
        self.edge_init = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + config.NUM_RBF, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # --- Backbone: DMPNN ---
        self.mp_layers = nn.ModuleList(
            [
                DMPNNLayer(self.hidden_dim, config.NUM_RBF, config.NUM_SBF)
                for _ in range(config.NUM_MP_LAYERS)
            ]
        )

        # --- Global Interaction ---
        # Project aggregated edges + atom embedding
        self.node_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.global_attention = GlobalAttention(
            hidden_dim=self.hidden_dim,
            n_heads=config.TRANSFORMER_HEADS,
            n_layers=config.TRANSFORMER_LAYERS,
            d_ff=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=0.1,  # Internal dropout allowed, readout is deterministic
        )

        # --- Readout ---
        # Embedding for missing edges (dist > cutoff)
        self.null_edge_emb = nn.Parameter(torch.zeros(1, self.hidden_dim))

        # Deterministic MLP (No Dropout)
        self.readout_mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        # Unpack batch
        atom_z = batch["atom_z"]
        edge_index = batch["edge_index"]  # (2, E)
        edge_rbf = batch["edge_rbf"]
        triplet_indices = batch["triplet_indices"]
        triplet_sbf = batch["triplet_sbf"]
        batch_atom = batch["batch_atom"]

        # 1. Initialize Embeddings
        x = self.atom_embedding(atom_z)  # (N, dim)

        # Edge initialization
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]
        x_dst = x[dst]

        edge_input = torch.cat([x_src, x_dst, edge_rbf], dim=1)
        h_edge = self.edge_init(edge_input)  # (E, dim)

        # 2. Local Geometric Encoding (DMPNN)
        for layer in self.mp_layers:
            h_edge = layer(h_edge, edge_rbf, triplet_sbf, triplet_indices)

        # 3. Global Interaction
        # Aggregate edge embeddings to nodes (incoming edges to target node)
        # dst is the target node index for edge (src -> dst)
        num_nodes = x.size(0)
        aggr_edge = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        aggr_edge.index_add_(0, dst, h_edge)

        # Combine atom info and local geometric context
        x_global_in = torch.cat([x, aggr_edge], dim=1)
        x_global_in = self.node_proj(x_global_in)

        # Apply Transformer
        x_global = self.global_attention(x_global_in, batch_atom)

        # 4. Fusion Readout
        c_atom_idx = batch["coupling_atom_index"]  # (C, 2)
        c_type = batch["coupling_type"]  # (C,)
        c_edge_idx = batch["coupling_edge_index"]  # (C,)

        # Gather global node features
        idx_0, idx_1 = c_atom_idx[:, 0], c_atom_idx[:, 1]
        feat_0 = x_global[idx_0]
        feat_1 = x_global[idx_1]

        # Gather local edge features
        # Handle missing edges (-1) by padding h_edge with a null embedding
        h_edge_padded = torch.cat([h_edge, self.null_edge_emb], dim=0)

        # Map -1 indices to the last row (null embedding)
        # We need a clone to not modify the original batch tensor in place if it's reused
        c_edge_idx_safe = c_edge_idx.clone()
        c_edge_idx_safe[c_edge_idx == -1] = h_edge.size(0)

        feat_edge = h_edge_padded[c_edge_idx_safe]

        # Type embedding
        feat_type = self.type_embedding(c_type)

        # Concatenate: [Global_u, Global_v, Local_uv, Type]
        readout_in = torch.cat([feat_0, feat_1, feat_edge, feat_type], dim=1)

        # Predict
        pred = self.readout_mlp(readout_in)

        return pred.squeeze(-1)


class Trainer:
    """
    Handles training, validation, and inference for HGANet.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = config.DEVICE
        self.scaler = TargetScaler()

    def setup_data(self):
        print("Setting up data...")
        # Train
        self.train_dataset = MolecularGraphDataset(
            self.config.TRAIN_CSV, "train", load_cached_data=True
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_graphs,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Validation
        self.val_dataset = MolecularGraphDataset(
            self.config.VAL_CSV, "val", load_cached_data=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Fit scaler on training data metadata
        # We need to load the dataframe for the scaler
        print("Fitting TargetScaler...")
        df_train = pd.read_csv(self.config.TRAIN_CSV)
        if self.config.DEBUG:
            df_train = df_train[
                df_train["molecule_name"].isin(
                    df_train["molecule_name"].unique()[: self.config.DEBUG_SAMPLE_SIZE]
                )
            ]
        self.scaler.fit(df_train)

    def train(self):
        self.setup_data()

        print("Initializing model...")
        model = HGANet(self.config).to(self.device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler: Linear Warmup + Cosine Annealing
        steps_per_epoch = len(self.train_loader)
        total_steps = self.config.MAX_EPOCHS * steps_per_epoch
        warmup_steps = self.config.WARMUP_EPOCHS * steps_per_epoch

        # Cite debug_lesson_12
        pct_start = warmup_steps / total_steps if total_steps > warmup_steps else 0.1

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

        criterion = nn.L1Loss()

        best_val_mae = float("inf")
        patience_counter = 0

        print(f"Starting training for {self.config.MAX_EPOCHS} epochs...")

        for epoch in range(1, self.config.MAX_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            start_time = time.time()

            for batch in self.train_loader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                optimizer.zero_grad()

                preds = model(batch)

                # Normalize targets
                targets = batch["coupling_value"]
                types = batch["coupling_type"].cpu().numpy()

                # We need to normalize targets on the fly or pre-normalize.
                # Since scaler uses numpy, we do it here.
                # Optimization: Could move this to dataset, but scaler needs full dataset stats.
                # We can use the scaler parameters to normalize tensor directly.

                # Manual normalization on GPU
                # Create mean/std tensors mapped by type
                # This is a bit slow to do every batch if not optimized, but robust.
                # Let's use the inverse of the logic: Predict normalized, calculate loss against normalized.

                # Get means/stds for types in this batch
                # We can construct a lookup tensor
                # types are 0..7
                type_means = torch.tensor(
                    [
                        self.scaler.stats[t]["mean"]
                        for t in sorted(self.scaler.stats.keys())
                    ],
                    device=self.device,
                )
                type_stds = torch.tensor(
                    [
                        self.scaler.stats[t]["std"]
                        for t in sorted(self.scaler.stats.keys())
                    ],
                    device=self.device,
                )

                # Map batch types to means/stds
                # Note: coupling_type in batch is integer index 0-7 if mapped correctly.
                # library.data.COUPLING_TYPES is sorted.
                # We assume library.data.TYPE_TO_ID matches the sorted keys of scaler.stats

                batch_means = type_means[batch["coupling_type"]]
                batch_stds = type_stds[batch["coupling_type"]]

                target_norm = (targets - batch_means) / batch_stds

                loss = criterion(preds, target_norm)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * batch["batch_size"]

            avg_train_loss = train_loss / len(self.train_dataset)

            # Validation
            val_mae_log = self.evaluate(model)
            val_score = val_mae_log["LMAE_Avg"]

            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch}/{self.config.MAX_EPOCHS} | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.6f} | Val LMAE: {val_score:.6f}"
            )

            # Checkpoint
            if val_score < best_val_mae:
                best_val_mae = val_score
                torch.save(model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  New best model saved! ({val_score:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        return model

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        # Type stats for denormalization
        type_means = torch.tensor(
            [self.scaler.stats[t]["mean"] for t in sorted(self.scaler.stats.keys())],
            device=self.device,
        )
        type_stds = torch.tensor(
            [self.scaler.stats[t]["std"] for t in sorted(self.scaler.stats.keys())],
            device=self.device,
        )

        for batch in self.val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            preds_norm = model(batch)

            # Denormalize predictions
            batch_means = type_means[batch["coupling_type"]]
            batch_stds = type_stds[batch["coupling_type"]]
            preds_raw = preds_norm * batch_stds + batch_means

            all_preds.append(preds_raw.cpu().numpy())
            all_targets.append(batch["coupling_value"].cpu().numpy())
            all_types.append(batch["coupling_type"].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_types = np.concatenate(all_types)

        # Calculate Log MAE per type
        # We need to map int types back to strings for reporting if needed,
        # but the metric is average of log(MAE) per type.

        metrics = {}
        log_maes = []

        # Get unique types present in validation
        unique_types = np.unique(all_types)

        for t_idx in unique_types:
            mask = all_types == t_idx
            mae = np.mean(np.abs(all_preds[mask] - all_targets[mask]))
            log_mae = np.log(mae + 1e-9)
            log_maes.append(log_mae)
            # metrics[f'LMAE_{t_idx}'] = log_mae

        metrics["LMAE_Avg"] = np.mean(log_maes)
        return metrics

    def predict_test(self):
        print("Loading best model for inference...")
        model = HGANet(self.config).to(self.device)
        model.load_state_dict(
            torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
        )
        model.eval()

        print("Processing test data...")
        test_dataset = MolecularGraphDataset(
            self.config.TEST_CSV, "test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=self.config.NUM_WORKERS,
        )

        # Type stats for denormalization
        # Ensure scaler is fit (it should be if train was called, else we need to load stats)
        if not self.scaler.fitted:
            # Try to fit on train metadata quickly
            df_train = pd.read_csv(self.config.TRAIN_CSV)
            self.scaler.fit(df_train)

        type_means = torch.tensor(
            [self.scaler.stats[t]["mean"] for t in sorted(self.scaler.stats.keys())],
            device=self.device,
        )
        type_stds = torch.tensor(
            [self.scaler.stats[t]["std"] for t in sorted(self.scaler.stats.keys())],
            device=self.device,
        )

        all_ids = []
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                preds_norm = model(batch)

                # Denormalize
                batch_means = type_means[batch["coupling_type"]]
                batch_stds = type_stds[batch["coupling_type"]]
                preds_raw = preds_norm * batch_stds + batch_means

                all_ids.append(batch["coupling_id"].cpu().numpy())
                all_preds.append(preds_raw.cpu().numpy())

        all_ids = np.concatenate(all_ids)
        all_preds = np.concatenate(all_preds)

        # Create submission dataframe
        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})
        df_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")


def run_training(config=Config):
    trainer = Trainer(config)
    trainer.train()
    trainer.predict_test()
