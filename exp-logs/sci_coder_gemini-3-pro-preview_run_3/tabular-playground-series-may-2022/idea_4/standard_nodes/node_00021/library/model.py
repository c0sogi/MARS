import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """
    Standard MLP Block: Linear -> BatchNorm -> ReLU -> Dropout
    Cite solution_lesson_node_00019: Simpler blocks often generalize better than GLU on this data.
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FunnelMLP(nn.Module):
    """
    Standard Funnel MLP Architecture.
    Combines high-capacity embeddings for categorical features with continuous features.
    Processes the combined input through a funnel of MLP Blocks.
    Cite solution_lesson_node_00004: Funnel architecture is superior for shallow tabular networks.
    """

    def __init__(self, vocab_sizes, cont_dim):
        super(FunnelMLP, self).__init__()

        # --- Embeddings ---
        self.embeddings = nn.ModuleList()
        self.cat_features = Config.CAT_FEATURES
        self.emb_dim = Config.EMBEDDING_DIM

        total_cat_dim = 0
        # Create embeddings in the order defined in Config.CAT_FEATURES
        for col in self.cat_features:
            num_embeddings = vocab_sizes[col]
            self.embeddings.append(nn.Embedding(num_embeddings, self.emb_dim))
            total_cat_dim += self.emb_dim

        # Total input dimension for the first dense layer
        input_dim = cont_dim + total_cat_dim

        # --- Funnel Layers ---
        self.layers = nn.ModuleList()
        hidden_dims = Config.HIDDEN_LAYERS
        dropout = Config.DROPOUT

        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(MLPBlock(current_dim, h_dim, dropout))
            current_dim = h_dim

        # --- Classification Head ---
        self.head = nn.Linear(current_dim, 1)

    def forward(self, cont_x, cat_x):
        """
        cont_x: Tensor of shape (batch_size, cont_dim)
        cat_x: Tensor of shape (batch_size, num_cat_features)
        """
        # Process Embeddings
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select the column corresponding to the i-th feature
            col_data = cat_x[:, i]
            emb_outputs.append(emb_layer(col_data))

        # Concatenate all embeddings: (batch_size, total_cat_dim)
        cat_out = torch.cat(emb_outputs, dim=1)

        # Concatenate with continuous features: (batch_size, input_dim)
        x = torch.cat([cont_x, cat_out], dim=1)

        # Pass through Funnel Layers
        for layer in self.layers:
            x = layer(x)

        # Final prediction (logits)
        logits = self.head(x)
        return logits


# -------------------------------------------------------------------------
# Training Logic
# -------------------------------------------------------------------------


def train_model(train_loader, val_loader, vocab_sizes, cont_dim):
    """
    Trains the FunnelMLP model using AdamW and OneCycleLR.
    Implements Early Stopping and saves the best model.
    """
    device = Config.DEVICE
    print(f"Initializing model on {device}...")

    model = FunnelMLP(vocab_sizes, cont_dim).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Loss Function (Binary Cross Entropy with Logits)
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping Tracking
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            cont_x = batch["cont_features"].to(device)
            cat_x = batch["cat_features"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(cont_x, cat_x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * cont_x.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                cont_x = batch["cont_features"].to(device)
                cat_x = batch["cat_features"].to(device)
                targets = batch["target"].to(device).unsqueeze(1)

                logits = model(cont_x, cat_x)
                loss = criterion(logits, targets)
                val_loss += loss.item() * cont_x.size(0)

                # Store for AUC calculation
                probs = torch.sigmoid(logits)
                all_targets.append(targets.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Calculate AUC
        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_preds)
        val_auc = roc_auc_score(y_true, y_pred)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    return model


# -------------------------------------------------------------------------
# Submission Logic
# -------------------------------------------------------------------------


def generate_submission(test_loader, vocab_sizes, cont_dim):
    """
    Loads the best model, predicts on the test set, and saves submission.csv.
    """
    device = Config.DEVICE
    print("Loading best model for inference...")

    # Re-instantiate model structure
    model = FunnelMLP(vocab_sizes, cont_dim).to(device)

    # Load weights
    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(
            "Error: Best model file not found. Ensure training completed successfully."
        )
        return

    model.eval()

    ids_list = []
    probs_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            cont_x = batch["cont_features"].to(device)
            cat_x = batch["cat_features"].to(device)
            ids = batch["id"]

            logits = model(cont_x, cat_x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            ids_list.extend(ids)
            probs_list.extend(probs)

    # Create DataFrame
    submission_df = pd.DataFrame({"id": ids_list, "target": probs_list})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
