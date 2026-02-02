import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_processing import DataHandler

logger = get_logger("model")


# -------------------------------------------------------------------------
# Custom Collate Function
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms in a batch.
    Pads atomic features to the maximum number of atoms in the batch and creates a mask.
    """
    ids = [item["id"] for item in batch]
    global_feats = torch.stack([item["global_features"] for item in batch])

    # Handle atomic features which vary in size (N_atoms, Features)
    atomic_feats_list = [item["atomic_features"] for item in batch]
    lengths = [x.shape[0] for x in atomic_feats_list]
    max_len = max(lengths)
    feature_dim = atomic_feats_list[0].shape[1]

    # Prepare padded tensor and mask
    # padded_atomic: (Batch, Max_N, Feat)
    # mask: (Batch, Max_N)
    padded_atomic = torch.zeros(len(batch), max_len, feature_dim)
    mask = torch.zeros(len(batch), max_len)

    for i, x in enumerate(atomic_feats_list):
        l = lengths[i]
        padded_atomic[i, :l, :] = x
        mask[i, :l] = 1.0

    targets = None
    if "targets" in batch[0]:
        targets = torch.stack([item["targets"] for item in batch])

    return {
        "ids": ids,
        "global_features": global_feats,
        "atomic_features": padded_atomic,
        "mask": mask,
        "targets": targets,
    }


# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------
class AtomicStream(nn.Module):
    """
    Wide Point Processor for atomic features.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.ATOMIC_INPUT_DIM, Config.ATOMIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.ATOMIC_DROPOUT),
            nn.Linear(Config.ATOMIC_HIDDEN_DIM, Config.ATOMIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.ATOMIC_DROPOUT),
            nn.Linear(Config.ATOMIC_HIDDEN_DIM, Config.ATOMIC_HIDDEN_DIM),
            # No activation on final projection
        )

    def forward(self, x):
        # x: (Batch, Max_N, Feat)
        return self.net(x)


class GlobalStream(nn.Module):
    """
    High-Capacity MLP for global thermodynamic context.
    """

    def __init__(self):
        super().__init__()
        layers = []
        input_dim = Config.GLOBAL_INPUT_DIM

        for _ in range(Config.GLOBAL_NUM_LAYERS):
            layers.append(nn.Linear(input_dim, Config.GLOBAL_HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.GLOBAL_DROPOUT))
            input_dim = Config.GLOBAL_HIDDEN_DIM

        # Final projection to embedding dimension
        layers.append(nn.Linear(input_dim, Config.GLOBAL_HIDDEN_DIM))
        layers.append(nn.ReLU())  # Activation before fusion

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, Global_Feat)
        return self.net(x)


class PAWDS(nn.Module):
    """
    Potential-Augmented Wide Deep Sets.
    """

    def __init__(self):
        super().__init__()
        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Input: (Atomic_Hidden * 2 for Mean+Max) + Global_Hidden
        fusion_input_dim = (Config.ATOMIC_HIDDEN_DIM * 2) + Config.GLOBAL_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.FUSION_HIDDEN_DIM, Config.OUTPUT_DIM),
        )

    def forward(self, atomic_x, global_x, mask):
        # atomic_x: (Batch, Max_N, Atom_Feat)
        # global_x: (Batch, Global_Feat)
        # mask: (Batch, Max_N)

        # 1. Atomic Stream
        # (Batch, Max_N, Atomic_Hidden)
        atomic_emb = self.atomic_stream(atomic_x)

        # Apply mask to embeddings (zero out padded atoms)
        mask_expanded = mask.unsqueeze(-1)  # (Batch, Max_N, 1)
        atomic_emb = atomic_emb * mask_expanded

        # 2. Dual Pooling
        # Mean Pooling: Sum / Count
        # Avoid division by zero
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = atomic_emb.sum(dim=1) / counts  # (Batch, Atomic_Hidden)

        # Max Pooling:
        # Replace 0s (padding) with -inf before max, but ensure we don't kill valid 0s if any.
        # Since ReLU is used in backbone, embeddings are non-negative? No, last layer is Linear.
        # So we must be careful.
        # Safe approach: fill padded positions with a very small number
        atomic_emb_for_max = atomic_emb.clone()
        # Invert mask: 1 where padding, 0 where valid
        padding_mask = (1.0 - mask).bool()
        atomic_emb_for_max[padding_mask] = -1e9
        max_pool = torch.max(atomic_emb_for_max, dim=1)[0]  # (Batch, Atomic_Hidden)

        # Concatenate pools
        atomic_agg = torch.cat(
            [mean_pool, max_pool], dim=1
        )  # (Batch, Atomic_Hidden * 2)

        # 3. Global Stream
        global_emb = self.global_stream(global_x)  # (Batch, Global_Hidden)

        # 4. Fusion
        fused = torch.cat([atomic_agg, global_emb], dim=1)  # (Batch, Fusion_Input)

        # 5. Prediction
        output = self.fusion_head(fused)  # (Batch, 2)

        return output


# -------------------------------------------------------------------------
# Training & Evaluation
# -------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        atomic_x = batch["atomic_features"].to(device)
        global_x = batch["global_features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(atomic_x, global_x, mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * atomic_x.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            atomic_x = batch["atomic_features"].to(device)
            global_x = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_x, global_x, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * atomic_x.size(0)

    return running_loss / len(loader.dataset)


def rmsle_metric(y_pred_log, y_true_log):
    """
    Calculate RMSLE.
    Since targets are already log(1+y), MSE on these targets is approximately MSLE.
    RMSLE = sqrt(MSE(log(1+y_pred), log(1+y_true)))
    """
    mse = np.mean((y_pred_log - y_true_log) ** 2)
    return np.sqrt(mse)


def run():
    # Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    data_handler = DataHandler()
    train_dataset, val_dataset, test_dataset = data_handler.get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Model
    model = PAWDS().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Time: {time.time() - start_time:.2f}s"
        )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # Inference
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_x = batch["atomic_features"].to(device)
            global_x = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["ids"]

            outputs = model(atomic_x, global_x, mask)

            # Inverse transform: exp(y) - 1
            preds_original_scale = torch.expm1(outputs).cpu().numpy()

            predictions.append(preds_original_scale)
            ids.extend(batch_ids)

    predictions = np.vstack(predictions)

    # Save Submission
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure columns are in correct order
    submission_df = submission_df[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
