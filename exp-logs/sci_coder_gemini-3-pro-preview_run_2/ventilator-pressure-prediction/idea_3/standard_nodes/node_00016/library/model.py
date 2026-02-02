import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.dataset import get_data_loaders
from library.utils import seed_everything, get_device, compute_metric


class HybridModel(nn.Module):
    """
    A Hybrid CNN-BiLSTM architecture for ventilator pressure prediction.

    Architecture:
    1. Embeddings for categorical lung attributes (R, C).
    2. 1D Convolutional Stem to extract local temporal dynamics.
    3. Stacked Bidirectional LSTM Body with Residual Connections to capture long-term dependencies.
    4. Dense Regression Head.
    """

    def __init__(
        self, input_dim=12, lstm_dim=256, num_lstm_layers=4, emb_dim=4, cnn_dim=256
    ):
        super().__init__()

        # --- Embeddings ---
        # R and C have cardinality 3 (mapped to 0, 1, 2)
        self.r_emb = nn.Embedding(3, emb_dim)
        self.c_emb = nn.Embedding(3, emb_dim)

        # Total input channels for CNN:
        # Continuous features (12) + R_emb (4) + C_emb (4)
        total_input_dim = input_dim + emb_dim * 2

        # --- CNN Stem ---
        # Extracts local features and projects to cnn_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(total_input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.GELU(),
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.GELU(),
        )

        # --- LSTM Body ---
        self.lstm_layers = nn.ModuleList()

        # First LSTM layer: Adapts CNN output to LSTM hidden size
        # Input: cnn_dim, Output: lstm_dim * 2 (Bidirectional)
        self.lstm_layers.append(
            nn.LSTM(cnn_dim, lstm_dim, batch_first=True, bidirectional=True)
        )

        # Subsequent LSTM layers: Residual connections possible
        # Input: lstm_dim * 2, Output: lstm_dim * 2
        for _ in range(num_lstm_layers - 1):
            self.lstm_layers.append(
                nn.LSTM(lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True)
            )

        # --- Regression Head ---
        self.head = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim), nn.GELU(), nn.Linear(lstm_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 14)
        # Features: [0..11] Continuous (incl u_out), [12] R_cat, [13] C_cat

        # 1. Feature Splitting
        x_cont = x[:, :, :-2]  # (B, L, 12)
        r_cat = x[:, :, -2].long()  # (B, L)
        c_cat = x[:, :, -1].long()  # (B, L)

        # 2. Embeddings
        r_emb = self.r_emb(r_cat)  # (B, L, emb_dim)
        c_emb = self.c_emb(c_cat)  # (B, L, emb_dim)

        # 3. Concatenation
        x_all = torch.cat([x_cont, r_emb, c_emb], dim=2)  # (B, L, 20)

        # 4. CNN Stem
        # Permute for Conv1d: (B, C, L)
        x_all = x_all.permute(0, 2, 1)
        x_cnn = self.cnn(x_all)
        # Permute back for LSTM: (B, L, C)
        x_lstm = x_cnn.permute(0, 2, 1)

        # 5. LSTM Body with Skip Connections
        for i, lstm in enumerate(self.lstm_layers):
            output, _ = lstm(x_lstm)

            # Apply residual connection if dimensions match (Layer 1 onwards)
            # Layer 0 output is (B, L, 2*H), Layer 1 input is (B, L, 2*H)
            if i > 0:
                x_lstm = output + x_lstm
            else:
                x_lstm = output

        # 6. Head
        pred = self.head(x_lstm)  # (B, L, 1)

        return pred.squeeze(-1)  # (B, L)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        X = batch["input"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()

        preds = model(X)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_u_out = []

    with torch.no_grad():
        for batch in loader:
            X = batch["input"].to(device)
            y = batch["target"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()

            all_preds.append(preds)
            all_targets.append(y)
            all_u_out.append(u_out)

    # Compute MAE on inspiratory phase only (competition metric)
    y_pred_cat = torch.cat(all_preds)
    y_true_cat = torch.cat(all_targets)
    u_out_cat = torch.cat(all_u_out)

    val_mae = compute_metric(y_pred_cat, y_true_cat, u_out_cat)

    return total_loss / len(loader), val_mae


def run_training_process(
    epochs=50, batch_size=256, lr=1e-3, debug=False, save_dir="./working/demo_execution"
):
    """
    Main function to execute the training pipeline and generate submission.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training process on {device}...")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size, load_cached_data=True, debug=debug
    )

    # 2. Initialize Model
    model = HybridModel(
        input_dim=12, lstm_dim=512, num_lstm_layers=4, emb_dim=8, cnn_dim=256
    ).to(device)

    # 3. Setup Optimization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    # Loss: L1 Loss over entire breath (inspiratory + expiratory)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # 4. Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE (Insp): {val_mae:.6f}"
        )

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MAE: {best_mae:.6f}")

    # 5. Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch["input"].to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy().flatten())

    final_preds = np.concatenate(all_preds)

    # 6. Create Submission File
    # We need to map predictions back to IDs.
    # dataset.py sorts test data by [breath_id, time_step].
    # We replicate this sort on metadata to get the correct ID order.
    test_meta_path = "./metadata/test_metadata.csv"
    if os.path.exists(test_meta_path):
        df_meta = pd.read_csv(test_meta_path)
        # Assuming id increments with time_step within a breath,
        # sorting by breath_id then id is equivalent to breath_id then time_step.
        df_meta = df_meta.sort_values(["breath_id", "id"])

        # Check alignment
        if len(df_meta) != len(final_preds):
            print(
                f"Warning: Prediction count {len(final_preds)} != Metadata count {len(df_meta)}"
            )
            # Truncate or pad if debug mode caused mismatch, though dataset.py handles debug slicing
            min_len = min(len(df_meta), len(final_preds))
            df_meta = df_meta.iloc[:min_len]
            final_preds = final_preds[:min_len]

        submission = pd.DataFrame({"id": df_meta["id"], "pressure": final_preds})

        sub_path = os.path.join(save_dir, "submission.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

        # Also save to the root submission folder if required by competition environment
        root_sub_dir = "./submission"
        os.makedirs(root_sub_dir, exist_ok=True)
        submission.to_csv(os.path.join(root_sub_dir, "submission.csv"), index=False)

    else:
        print("Error: Test metadata not found. Cannot generate submission with IDs.")
