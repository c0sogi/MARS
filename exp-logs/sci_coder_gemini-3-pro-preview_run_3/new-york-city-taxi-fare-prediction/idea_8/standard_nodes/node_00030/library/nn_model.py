import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# ==========================================
# 1. DATASET
# ==========================================


class TaxiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode="train"):
        """
        PyTorch Dataset for Taxi Fare Prediction.

        Args:
            df (pd.DataFrame): The dataframe containing features and target.
            mode (str): 'train', 'val', or 'test'. If 'test', target is not expected.
        """
        self.mode = mode

        # 1. Continuous Features
        # Combine standard continuous features and cyclical features
        self.cont_cols = config.NN_CONTINUOUS_FEATURES + config.NN_CYCLICAL_FEATURES

        # Ensure all columns exist
        missing_cont = [c for c in self.cont_cols if c not in df.columns]
        if missing_cont:
            raise ValueError(f"Missing continuous columns: {missing_cont}")

        self.cont_data = df[self.cont_cols].values.astype(np.float32)

        # 2. Categorical Features (for Embeddings)
        self.cat_data = []
        self.cat_cols = [c[0] for c in config.NN_EMBEDDING_CONFIG]

        if self.cat_cols:
            # Ensure columns exist
            missing_cat = [c for c in self.cat_cols if c not in df.columns]
            if missing_cat:
                raise ValueError(f"Missing categorical columns: {missing_cat}")

            # Stack categorical columns (N, num_cats)
            self.cat_data = df[self.cat_cols].values.astype(np.int64)

        # 3. Target
        if mode != "test":
            if "fare_amount" not in df.columns:
                raise ValueError("Target 'fare_amount' missing in train/val dataset")
            self.target = df["fare_amount"].values.astype(np.float32)
        else:
            self.target = None

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        # Continuous features
        x_cont = torch.tensor(self.cont_data[idx], dtype=torch.float32)

        # Categorical features
        x_cat = (
            torch.tensor(self.cat_data[idx], dtype=torch.long)
            if len(self.cat_cols) > 0
            else torch.tensor([], dtype=torch.long)
        )

        if self.mode != "test":
            y = torch.tensor(self.target[idx], dtype=torch.float32)
            return x_cont, x_cat, y
        else:
            return x_cont, x_cat


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()

        # Main path: Linear -> BN -> ReLU -> Dropout
        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shortcut path: Projection if dimensions change, else Identity
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

        # Final activation after addition (optional but common, here we follow prompt structure ending in Add)
        # Prompt: "Linear -> BatchNorm -> ReLU -> Dropout -> Residual Add"
        # We will return the sum.

    def forward(self, x):
        out = self.main_path(x)
        res = self.shortcut(x)
        return out + res


class TaxiResNet(nn.Module):
    def __init__(self):
        super(TaxiResNet, self).__init__()

        # 1. Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, emb_dim)
                for _, num_classes, emb_dim in config.NN_EMBEDDING_CONFIG
            ]
        )

        total_emb_dim = sum(emb_dim for _, _, emb_dim in config.NN_EMBEDDING_CONFIG)
        num_cont = len(config.NN_CONTINUOUS_FEATURES) + len(config.NN_CYCLICAL_FEATURES)

        input_dim = num_cont + total_emb_dim

        # 2. Hidden Layers (Residual Blocks)
        hidden_dims = config.RESNET_PARAMS["hidden_dims"]
        dropout = config.RESNET_PARAMS["dropout"]

        layers = []

        # Initial projection to first hidden dimension
        # We treat this as the first block or a projection before blocks
        # To strictly follow ResNet, we project input to hidden_dim[0] then apply blocks
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )

        # Stack Residual Blocks
        # We iterate through hidden_dims.
        # If we want to reduce dimension, we do it at the block level.
        current_dim = hidden_dims[0]
        for next_dim in hidden_dims:
            layers.append(ResidualBlock(current_dim, next_dim, dropout))
            current_dim = next_dim

        self.blocks = nn.Sequential(*layers)

        # 3. Output Head
        self.output_head = nn.Linear(current_dim, 1)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_cont, x_cat):
        # Process Embeddings
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] is the column for the i-th categorical feature
            emb_outputs.append(emb_layer(x_cat[:, i]))

        if emb_outputs:
            x_emb = torch.cat(emb_outputs, dim=1)
            x = torch.cat([x_cont, x_emb], dim=1)
        else:
            x = x_cont

        # Project
        x = self.input_proj(x)

        # Residual Blocks
        x = self.blocks(x)

        # Output
        out = self.output_head(x)
        return out.squeeze(-1)


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================


def train_resnet(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Trains the TaxiResNet model.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.

    Returns:
        tuple: (trained_model, val_predictions)
    """
    utils.seed_everything(config.SEED)
    device = utils.get_device()

    print(f"\n=== Training Deep Spatial ResNet ===")
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    print(f"Device: {device}")

    # Hyperparameters
    params = config.RESNET_PARAMS
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    lr = params["learning_rate"]
    patience = params["patience"]

    # Datasets & Loaders
    train_dataset = TaxiDataset(train_df, mode="train")
    val_dataset = TaxiDataset(val_df, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.N_JOBS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=config.N_JOBS,
        pin_memory=True,
    )

    # Model Setup
    model = TaxiResNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=params["weight_decay"]
    )
    loss_fn = nn.MSELoss()

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    # Training Loop
    best_rmse = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x_cont, x_cat)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * x_cont.size(0)

        train_loss /= len(train_dataset)
        train_rmse = np.sqrt(train_loss)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
                preds = model(x_cont, x_cat)
                loss = loss_fn(preds, y)
                val_loss += loss.item() * x_cont.size(0)

                val_preds.append(preds.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_rmse = np.sqrt(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}"
        )

        # Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save checkpoint
            os.makedirs(os.path.dirname(config.MODEL_RESNET_PATH), exist_ok=True)
            torch.save(best_model_state, config.MODEL_RESNET_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation RMSE: {best_rmse:.5f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Generate full validation predictions for stacking
    model.eval()
    all_val_preds = []
    with torch.no_grad():
        for x_cont, x_cat, _ in val_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            preds = model(x_cont, x_cat)
            all_val_preds.append(preds.cpu().numpy())

    return model, np.concatenate(all_val_preds)


# ==========================================
# 4. INFERENCE FUNCTION
# ==========================================


def predict_resnet(model: nn.Module, test_df: pd.DataFrame):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): Trained TaxiResNet model.
        test_df (pd.DataFrame): Test data.

    Returns:
        np.array: Predictions.
    """
    device = utils.get_device()
    model.eval()
    model.to(device)

    dataset = TaxiDataset(test_df, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=config.RESNET_PARAMS["batch_size"] * 2,
        shuffle=False,
        num_workers=config.N_JOBS,
    )

    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat in loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            preds = model(x_cont, x_cat)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)
