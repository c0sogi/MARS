import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import masked_mae_loss, get_device


class DCLKNet(nn.Module):
    """
    Dense-Context Large-Kernel Hybrid (DCLK-Net)

    A parallel hybrid architecture combining:
    1. A deep, dense, large-kernel 1D CNN stack for high-frequency resistive dynamics.
    2. A high-capacity bidirectional LSTM for low-frequency elastic integration.
    """

    def __init__(self):
        super(DCLKNet, self).__init__()

        # --- Hyperparameters ---
        self.input_dim = 14  # Derived from feature engineering pipeline

        # CNN Branch Config
        self.cnn_layers = Config.CNN_LAYERS
        self.cnn_kernel = Config.CNN_KERNEL_SIZE
        self.cnn_dilation = Config.CNN_DILATION
        self.cnn_start = Config.CNN_CHANNELS_START
        self.cnn_max = Config.CNN_CHANNELS_MAX

        # LSTM Branch Config
        self.lstm_layers = Config.LSTM_LAYERS
        self.lstm_hidden = Config.LSTM_HIDDEN
        self.lstm_bidir = Config.LSTM_BIDIRECTIONAL

        # General Config
        self.dropout_p = Config.DROPOUT

        # --- Branch 1: Deep Dense Large-Kernel TCN (Resistive Stream) ---
        self.cnn_branch = nn.ModuleList()
        current_channels = self.input_dim

        # Calculate centered padding for stride=1
        # padding = (dilation * (kernel - 1)) / 2
        padding = (self.cnn_dilation * (self.cnn_kernel - 1)) // 2

        for i in range(self.cnn_layers):
            # Channel capacity increases with depth until max
            out_channels = min(self.cnn_start * (2**i), self.cnn_max)

            layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=self.cnn_kernel,
                    stride=1,
                    padding=padding,
                    dilation=self.cnn_dilation,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
            self.cnn_branch.append(layer)
            current_channels = out_channels

        self.cnn_out_dim = current_channels

        # --- Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream) ---
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.lstm_bidir,
        )

        self.lstm_out_dim = (
            self.lstm_hidden * 2 if self.lstm_bidir else self.lstm_hidden
        )

        # --- Fusion Head ---
        # Concatenation of CNN and LSTM outputs
        fusion_dim = self.cnn_out_dim + self.lstm_out_dim

        # Deep Dense MLP without skip connections
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Explicit weight initialization to ensure stability.
        Cite solution_lesson_node_00065
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (Batch, Length, Features)
        Returns:
            torch.Tensor: Output of shape (Batch, Length)
        """
        # 1. CNN Branch
        # Permute to (Batch, Channels, Length) for Conv1d
        x_cnn = x.permute(0, 2, 1)
        for layer in self.cnn_branch:
            x_cnn = layer(x_cnn)
        # Permute back to (Batch, Length, Channels)
        x_cnn = x_cnn.permute(0, 2, 1)

        # 2. LSTM Branch
        # Input is already (Batch, Length, Features)
        x_lstm, _ = self.lstm(x)

        # 3. Fusion
        # Concatenate along the feature dimension
        x_fused = torch.cat([x_cnn, x_lstm], dim=2)

        # 4. Head
        out = self.head(x_fused)

        # Squeeze the last dimension: (Batch, Length, 1) -> (Batch, Length)
        return out.squeeze(-1)


def train_model(train_loader, val_loader):
    """
    Trains the DCLKNet model with early stopping and learning rate scheduling.

    Args:
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.

    Returns:
        DCLKNet: The trained model with the best validation weights loaded.
    """
    device = get_device()
    model = DCLKNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_sum = 0
        train_batches = 0

        for x, u_out, y, _ in train_loader:
            x, u_out, y = x.to(device), u_out.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = masked_mae_loss(preds, y, u_out)

            loss.backward()
            # Gradient Clipping is mandatory for hybrid architectures (Cite solution_lesson_node_00065)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        val_batches = 0

        with torch.no_grad():
            for x, u_out, y, _ in val_loader:
                x, u_out, y = x.to(device), u_out.to(device), y.to(device)
                preds = model(x)
                loss = masked_mae_loss(preds, y, u_out)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches

        # --- Updates ---
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Loss: {best_val_loss:.6f}")

    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict_and_submit(model, test_loader):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (DCLKNet): Trained model.
        test_loader (DataLoader): Test data.
    """
    device = get_device()
    model.eval()

    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for x, u_out, _, ids in test_loader:
            x = x.to(device)

            preds = model(x)

            # Move to CPU and flatten
            preds_np = preds.cpu().numpy().flatten()
            ids_np = ids.numpy().flatten()

            all_preds.append(preds_np)
            all_ids.append(ids_np)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    # Create DataFrame
    submission = pd.DataFrame({Config.ID_COL: all_ids, Config.TARGET_COL: all_preds})

    # Ensure correct sorting
    submission.sort_values(by=Config.ID_COL, inplace=True)

    # Save
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
