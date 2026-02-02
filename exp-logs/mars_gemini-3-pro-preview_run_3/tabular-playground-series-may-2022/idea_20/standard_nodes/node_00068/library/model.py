import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from library.config import Config

# -------------------------------------------------------------------------
# Building Blocks
# -------------------------------------------------------------------------


class FunnelBlock(nn.Module):
    """
    Standard MLP Block: Linear -> BN -> Activation -> Dropout.
    Supports ReLU and SiLU (Swish).
    """

    def __init__(
        self, in_features, out_features, activation_name="ReLU", dropout_rate=0.2
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

        if activation_name == "SiLU":
            self.activation = nn.SiLU()
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GLUBlock(nn.Module):
    """
    Gated Linear Unit Block: Linear -> BN -> GLU -> Dropout.
    The Linear layer projects to 2 * out_features, and GLU reduces it back to out_features.
    """

    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)
        self.bn = nn.BatchNorm1d(out_features * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.glu(x)
        x = self.dropout(x)
        return x


class StreamModule(nn.Module):
    """
    Independent stream containing its own embeddings, backbone, and output head.
    """

    def __init__(self, vocab_sizes, num_cont, embed_dim, stream_config):
        super().__init__()

        # 1. Independent Embeddings
        # Each stream learns its own representation of the categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Calculate input dimension for the backbone
        # (Num_Cat * Embed_Dim) + Num_Cont
        self.input_dim = (len(vocab_sizes) * embed_dim) + num_cont

        # 2. Backbone Construction
        layers = []
        in_dim = self.input_dim
        hidden_dims = stream_config["layers"]
        dropout = stream_config["dropout"]
        block_type = stream_config["type"]
        act_name = stream_config["act"]

        for h_dim in hidden_dims:
            if block_type == "glu":
                layers.append(GLUBlock(in_dim, h_dim, dropout_rate=dropout))
            else:
                layers.append(
                    FunnelBlock(
                        in_dim, h_dim, activation_name=act_name, dropout_rate=dropout
                    )
                )
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Head (Binary Classification)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_cont, x_cat):
        # Embed categorical features
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] is the column of indices for the i-th categorical feature
            val = emb_layer(x_cat[:, i])
            embedded.append(val)

        # Concatenate embeddings: [Batch, Num_Cat * Embed_Dim]
        cat_features = torch.cat(embedded, dim=1)

        # Early Fusion: Concatenate with continuous features
        x = torch.cat([cat_features, x_cont], dim=1)

        # Pass through backbone
        x = self.backbone(x)

        # Output Logits
        logits = self.head(x)
        return logits


# -------------------------------------------------------------------------
# Main Model Architecture
# -------------------------------------------------------------------------


class SDPEModel(nn.Module):
    """
    Structurally Diverse Parallel Ensemble (SDPE).
    Contains 5 independent streams defined in Config.STREAM_CONFIGS.
    """

    def __init__(self, vocab_sizes, num_cont):
        super().__init__()

        self.streams = nn.ModuleList()

        for config in Config.STREAM_CONFIGS:
            stream = StreamModule(
                vocab_sizes=vocab_sizes,
                num_cont=num_cont,
                embed_dim=Config.EMBED_DIM,
                stream_config=config,
            )
            self.streams.append(stream)

    def forward(self, x_cont, x_cat):
        """
        Forward pass runs all streams in parallel.
        Returns:
            list[torch.Tensor]: A list of 5 tensors (logits), one from each stream.
        """
        outputs = []
        for stream in self.streams:
            logits = stream(x_cont, x_cat)
            outputs.append(logits)

        return outputs


# -------------------------------------------------------------------------
# Training and Inference Functions
# -------------------------------------------------------------------------


def train_sdpe_model(model, train_loader, val_loader):
    """
    Trains the SDPE model using the specified strategy:
    - Optimizer: Adam (Standard)
    - Scheduler: OneCycleLR
    - Loss: Sum of BCE from all streams
    - Early Stopping based on Validation AUC
    """
    device = Config.DEVICE
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass returns list of logits
            outputs = model(x_cont, x_cat)

            # Loss is sum of BCEs
            loss = 0
            for logits in outputs:
                loss += criterion(logits, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x_cont = batch["x_cont"].to(device)
                x_cat = batch["x_cat"].to(device)
                target = batch["target"].to(device)

                outputs = model(x_cont, x_cat)

                # Ensemble Mean for Validation
                # Stack logits: [5, Batch, 1]
                # Sigmoid -> [5, Batch, 1]
                # Mean -> [Batch, 1]
                probs = torch.stack([torch.sigmoid(out) for out in outputs])
                avg_prob = torch.mean(probs, dim=0).squeeze(1)

                val_preds.extend(avg_prob.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {avg_train_loss} - Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with AUC: {val_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def predict_sdpe_model(model, test_loader):
    """
    Generates predictions using the trained model and saves to submission.csv.
    Uses Ensemble Mean of the 5 streams.
    """
    device = Config.DEVICE
    model.to(device)

    # Load best weights
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading best model from {Config.MODEL_PATH}...")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model file not found. Using current weights.")

    model.eval()
    all_ids = []
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            ids = batch["id"]

            outputs = model(x_cont, x_cat)

            # Ensemble Mean
            probs = torch.stack([torch.sigmoid(out) for out in outputs])
            avg_prob = torch.mean(probs, dim=0).squeeze(1)

            all_preds.extend(avg_prob.cpu().numpy())
            all_ids.extend(ids.numpy())

    # Create submission dataframe
    df_sub = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Ensure ID is int
    df_sub["id"] = df_sub["id"].astype(int)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
