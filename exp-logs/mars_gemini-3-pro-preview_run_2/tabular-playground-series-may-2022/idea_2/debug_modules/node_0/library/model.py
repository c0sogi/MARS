import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

from library.config import (
    NUM_CONTINUOUS_FEATURES,
    NUM_CATEGORICAL_POSITIONS,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    DEEP_HIDDEN_UNITS,
    DEEP_DROPOUT,
    NUM_CROSS_LAYERS,
    MODEL_SAVE_PATH,
    SUBMISSION_SAVE_PATH,
    TEST_META_PATH,
    DEVICE,
)

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------


class CrossLayer(nn.Module):
    """
    Implements a single layer of the Cross Network.
    Formula: x_{l+1} = x_0 * (x_l^T . w_l) + b_l + x_l
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim

        # Weight w_l: (input_dim, 1)
        self.weight = nn.Parameter(torch.empty(input_dim, 1))
        # Bias b_l: (input_dim, )
        self.bias = nn.Parameter(torch.empty(input_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input feature vector (Batch, Input_Dim)
            xl: Output from previous cross layer (Batch, Input_Dim)
        Returns:
            x_next: Output of this cross layer (Batch, Input_Dim)
        """
        # Compute scalar interaction (x_l^T . w_l) for each sample in batch
        # x_l: (B, D), weight: (D, 1) -> matmul -> (B, 1)
        interaction = torch.matmul(xl, self.weight)

        # Apply formula: x_0 * interaction + bias + x_l
        # Broadcasting interaction (B, 1) to (B, D)
        x_next = x0 * interaction + self.bias + xl
        return x_next


class ParallelDCN(nn.Module):
    """
    Parallel Deep & Cross Network (DCN-V2) architecture.
    Combines a Deep (MLP) branch and a Cross branch in parallel.
    """

    def __init__(
        self,
        num_continuous=NUM_CONTINUOUS_FEATURES,
        num_categorical_pos=NUM_CATEGORICAL_POSITIONS,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        deep_hidden_units=DEEP_HIDDEN_UNITS,
        deep_dropout=DEEP_DROPOUT,
        num_cross_layers=NUM_CROSS_LAYERS,
    ):
        super(ParallelDCN, self).__init__()

        # 1. Input Processing
        # Shared embedding layer for all character positions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Calculate total dense input dimension
        # Flattened embeddings + continuous features
        self.input_dim = (num_categorical_pos * embedding_dim) + num_continuous

        # 2. Cross Network Branch
        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.input_dim) for _ in range(num_cross_layers)]
        )

        # 3. Deep Network Branch (MLP)
        deep_layers = []
        in_dim = self.input_dim
        for hidden_dim in deep_hidden_units:
            deep_layers.append(nn.Linear(in_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(deep_dropout))
            in_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)
        self.deep_out_dim = deep_hidden_units[-1]

        # 4. Fusion Head
        # Concatenate Cross output (input_dim) and Deep output (deep_out_dim)
        stack_dim = self.input_dim + self.deep_out_dim
        self.final_linear = nn.Linear(stack_dim, 1)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: Categorical indices (Batch, 10)
            x_cont: Continuous features (Batch, 30)
        """
        # Embed and Flatten Categorical
        # (B, 10) -> (B, 10, Embed) -> (B, 10 * Embed)
        bs = x_cat.size(0)
        embeds = self.embedding(x_cat).view(bs, -1)

        # Concatenate with Continuous -> x0
        x0 = torch.cat([embeds, x_cont], dim=1)

        # Cross Branch
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        cross_out = xl

        # Deep Branch
        deep_out = self.deep_net(x0)

        # Fusion
        stacked = torch.cat([cross_out, deep_out], dim=1)
        logits = self.final_linear(stacked)

        return logits


# ------------------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------------------


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        cat_feats = batch["cat"].to(device)
        cont_feats = batch["cont"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward
        logits = model(cat_feats, cont_feats)
        loss = criterion(logits, targets)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * cat_feats.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cat_feats = batch["cat"].to(device)
            cont_feats = batch["cont"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            logits = model(cat_feats, cont_feats)
            loss = criterion(logits, targets)

            running_loss += loss.item() * cat_feats.size(0)

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    epoch_loss = running_loss / len(dataloader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    patience,
    save_path=MODEL_SAVE_PATH,
):
    """
    Full training loop with Early Stopping based on Validation AUC.
    """
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # print(f"  -> Model saved! New Best AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            # print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # Load best model state
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ------------------------------------------------------------------------------
# Inference & Submission
# ------------------------------------------------------------------------------


def predict(model, dataloader, device):
    """
    Generates probabilities for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cat_feats = batch["cat"].to(device)
            cont_feats = batch["cont"].to(device)

            logits = model(cat_feats, cont_feats)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.vstack(all_preds).flatten()


def generate_submission(model, test_loader, device, output_path=SUBMISSION_SAVE_PATH):
    """
    Generates predictions and saves them to a CSV file.
    """
    print("Generating predictions for test set...")
    probs = predict(model, test_loader, device)

    # Load Test Metadata to get correct IDs
    if not os.path.exists(TEST_META_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_META_PATH}")

    df_test_meta = pd.read_csv(TEST_META_PATH)

    # Ensure lengths match
    if len(probs) != len(df_test_meta):
        print(
            f"Warning: Prediction count ({len(probs)}) does not match Test ID count ({len(df_test_meta)})."
        )

    # Create submission dataframe
    submission_df = pd.DataFrame({"id": df_test_meta["id"], "target": probs})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
