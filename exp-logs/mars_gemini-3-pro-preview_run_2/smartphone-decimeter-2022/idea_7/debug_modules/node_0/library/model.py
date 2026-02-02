import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from library.config import Config
from library.utils import meters_to_deg


class LocalAttentionTransformer(nn.Module):
    """
    Local-Attention Transformer (LAT) for GNSS trajectory refinement.

    Input: (Batch, Window_Size, Input_Dim)
    Output: (Batch, 2) -> [Delta_Lat_Meters, Delta_Lon_Meters]
    """

    def __init__(self):
        super(LocalAttentionTransformer, self).__init__()

        self.d_model = Config.D_MODEL
        self.input_dim = Config.INPUT_DIM
        self.output_dim = Config.OUTPUT_DIM
        self.window_size = Config.WINDOW_SIZE
        self.nhead = Config.NHEAD
        self.num_layers = Config.NUM_ENCODER_LAYERS
        self.dim_feedforward = Config.DIM_FEEDFORWARD
        self.dropout_p = Config.DROPOUT

        # 1. Input Embedding
        self.embedding = nn.Linear(self.input_dim, self.d_model)

        # 2. Learnable Positional Encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.window_size, self.d_model)
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # 4. Prediction Head
        # Predicts residuals (meters) from the center token
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.dim_feedforward, self.dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.dim_feedforward // 2, self.output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (Batch, Window, Features)

        # Embed and add position
        x = self.embedding(x)  # (Batch, Window, D_Model)
        x = x + self.pos_embedding

        # Pass through Transformer
        x = self.transformer_encoder(x)  # (Batch, Window, D_Model)

        # Extract Center Token
        # The window is centered on the target epoch.
        center_idx = self.window_size // 2
        center_feat = x[:, center_idx, :]  # (Batch, D_Model)

        # Predict
        out = self.head(center_feat)  # (Batch, 2)
        return out


def train_model(train_loader, val_loader):
    """
    Trains the LocalAttentionTransformer model with Early Stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = LocalAttentionTransformer().to(device)

    # Robust L1 Loss for outliers
    criterion = nn.L1Loss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False
            ):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.10f}, Val Loss = {avg_val_loss:.10f}"
        )

        # Learning Rate Scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    return model


def predict_and_submit(test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train model first."
        )

    model = LocalAttentionTransformer().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch_x, meta in tqdm(test_loader, desc="Inference"):
            batch_x = batch_x.to(device)

            # Predict residuals (meters)
            # Output: [d_lat_m, d_lon_m]
            preds_m = model(batch_x).cpu().numpy()

            # Metadata for reconstruction
            trip_ids = meta["tripId"]
            timestamps = meta["UnixTimeMillis"].numpy()
            wls_lats = meta["wls_lat"].numpy()
            wls_lons = meta["wls_lon"].numpy()

            # Convert metric residuals to degrees
            d_lat_deg, d_lon_deg = meters_to_deg(preds_m[:, 0], preds_m[:, 1], wls_lats)

            # Apply correction
            pred_lats = wls_lats + d_lat_deg
            pred_lons = wls_lons + d_lon_deg

            # Store results
            for i in range(len(trip_ids)):
                results.append(
                    {
                        "tripId": trip_ids[i],
                        "UnixTimeMillis": timestamps[i],
                        "LatitudeDegrees": pred_lats[i],
                        "LongitudeDegrees": pred_lons[i],
                    }
                )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission_df = submission_df[cols]

    # Save
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print("Submission saved successfully.")
