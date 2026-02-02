import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, meters_to_latlon
from library.data_loader import load_data, GNSSWindowDataset

# Initialize logger
logger = get_logger("model")


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        # Shape: (1, max_len, d_model) for batch broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # Slice pe to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SkyStateTransformer(nn.Module):
    """
    Dual-Stream Transformer for Anchor-Free Trajectory Refinement.

    Stream 1: Trajectory Stream (Transformer)
        - Processes a sequence of relative positions, velocities, and signal metrics.
        - Uses self-attention to weigh temporal neighbors.
        - Extracts the embedding of the CENTER timestamp.

    Stream 2: Sky-State Context Stream (MLP)
        - Processes aggregated statistics of the satellite geometry and signal quality.
        - Learns an environmental bias embedding (e.g., Open Sky vs. Urban Canyon).

    Fusion:
        - Concatenates the Center Token and Sky Embedding.
        - Projects to 2D metric residuals (Delta East, Delta North).
    """

    def __init__(self):
        super(SkyStateTransformer, self).__init__()

        # --- Hyperparameters ---
        self.seq_input_dim = len(Config.SEQ_FEATURES)
        self.sky_input_dim = len(Config.SKY_FEATURES)
        self.d_model = Config.TRANSFORMER_HIDDEN_SIZE
        self.nhead = Config.TRANSFORMER_NUM_HEADS
        self.num_layers = Config.TRANSFORMER_NUM_LAYERS
        self.dropout_p = Config.TRANSFORMER_DROPOUT
        self.sky_embed_dim = Config.SKY_EMBED_SIZE
        self.mlp_hidden = Config.MLP_HIDDEN_SIZE
        self.output_dim = Config.OUTPUT_DIM
        self.window_size = Config.WINDOW_SIZE

        # --- Trajectory Stream ---
        # Project input features to d_model
        self.seq_embedding = nn.Linear(self.seq_input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=self.window_size, dropout=self.dropout_p
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout_p,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # --- Sky-State Context Stream ---
        self.sky_mlp = nn.Sequential(
            nn.Linear(self.sky_input_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.mlp_hidden, self.sky_embed_dim),
            nn.ReLU(),
        )

        # --- Fusion Head ---
        fusion_input_dim = self.d_model + self.sky_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.mlp_hidden, self.output_dim),
        )

    def forward(self, x_seq, x_sky):
        """
        Args:
            x_seq: (Batch, Window, Seq_Feats)
            x_sky: (Batch, Sky_Feats)
        Returns:
            output: (Batch, 2) -> [Delta East, Delta North] in meters
        """
        # 1. Trajectory Stream
        # Embed sequence
        seq_emb = self.seq_embedding(x_seq)  # (Batch, Window, d_model)
        # Add positional encoding
        seq_emb = self.pos_encoder(seq_emb)
        # Pass through Transformer
        trans_out = self.transformer_encoder(seq_emb)  # (Batch, Window, d_model)

        # Extract Center Token (The target timestamp is at the middle of the window)
        center_idx = self.window_size // 2
        center_token = trans_out[:, center_idx, :]  # (Batch, d_model)

        # 2. Sky-State Stream
        sky_emb = self.sky_mlp(x_sky)  # (Batch, sky_embed_dim)

        # 3. Fusion
        fused = torch.cat(
            [center_token, sky_emb], dim=1
        )  # (Batch, d_model + sky_embed_dim)
        output = self.fusion_head(fused)  # (Batch, 2)

        return output


def train_model():
    """
    Orchestrates the training process: loads data, initializes model, runs training loop.
    """
    logger.info("Starting model training...")

    # 1. Load Data
    # load_data handles caching automatically
    (train_data, val_data, _) = load_data(load_cached_data=True)

    train_X_seq, train_X_sky, train_y = train_data
    val_X_seq, val_X_sky, val_y, val_meta = val_data

    # Create Datasets and Loaders
    train_dataset = GNSSWindowDataset(train_X_seq, train_X_sky, train_y)
    val_dataset = GNSSWindowDataset(val_X_seq, val_X_sky, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    Config.set_seed()
    device = torch.device(Config.DEVICE)
    model = SkyStateTransformer().to(device)

    # 3. Optimization
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(
        f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples."
    )

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for batch_seq, batch_sky, batch_y in train_loader:
            batch_seq = batch_seq.to(device)
            batch_sky = batch_sky.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_seq, batch_sky)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_seq.size(0)

        avg_train_loss = train_loss_sum / len(train_dataset)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_seq, batch_sky, batch_y in val_loader:
                batch_seq = batch_seq.to(device)
                batch_sky = batch_sky.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_seq, batch_sky)
                loss = criterion(outputs, batch_y)
                val_loss_sum += loss.item() * batch_seq.size(0)

        avg_val_loss = val_loss_sum / len(val_dataset)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
        )

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info(f"New best model saved with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")


def generate_submission():
    """
    Generates predictions for the test set and saves the submission file.
    """
    logger.info("Generating submission...")

    # 1. Load Data
    (_, _, test_data) = load_data(load_cached_data=True)
    test_X_seq, test_X_sky, test_meta = test_data

    test_dataset = GNSSWindowDataset(test_X_seq, test_X_sky, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = SkyStateTransformer().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train the model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 3. Predict
    all_preds = []
    with torch.no_grad():
        for batch_seq, batch_sky in test_loader:
            batch_seq = batch_seq.to(device)
            batch_sky = batch_sky.to(device)

            outputs = model(batch_seq, batch_sky)
            all_preds.append(outputs.cpu().numpy())

    # Shape: (N_test, 2) -> [Delta East, Delta North]
    predictions_meters = np.concatenate(all_preds, axis=0)

    # 4. Reconstruction
    # We need to convert metric residuals back to Lat/Lon and add to WLS baseline
    # test_meta contains 'WlsLat', 'WlsLon'

    wls_lat = test_meta["WlsLat"].values
    wls_lon = test_meta["WlsLon"].values

    delta_east = predictions_meters[:, 0]
    delta_north = predictions_meters[:, 1]

    # Convert meters to lat/lon offsets
    pred_lat, pred_lon = meters_to_latlon(delta_north, delta_east, wls_lat, wls_lon)

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
