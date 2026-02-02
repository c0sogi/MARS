import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config


class ShallowEmbeddingMLP(nn.Module):
    """
    A shallow feed-forward neural network that combines learned embeddings for 
    categorical sequences with continuous features.
    
    Architecture:
    [Categorical Input] -> Embedding -> Flatten --\
                                                  +-> Concat -> Dense -> ReLU -> BN -> Dropout -> Dense -> Sigmoid
    [Continuous Input] --------------------------/
    """

    def __init__(self):
        super(ShallowEmbeddingMLP, self).__init__()

        # Hyperparameters from Config
        self.vocab_size = Config.VOCAB_SIZE
        self.embedding_dim = Config.EMBEDDING_DIM
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.continuous_dim = len(Config.CONTINUOUS_FEATURES)
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_prob = Config.DROPOUT

        # 1. Embedding Layer for Categorical Data
        # Input: (Batch, Sequence_Length) -> Output: (Batch, Sequence_Length, Embedding_Dim)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )

        # Calculate the dimension after flattening embeddings and concatenating continuous features
        # Flattened Embedding Size = 10 * 16 = 160
        # Total Input Size = 160 + 30 = 190
        self.concat_dim = (
            self.sequence_length * self.embedding_dim
        ) + self.continuous_dim

        # 2. Hidden Layer Block
        # Deepening the network to capture interactions between embeddings and continuous features
        # Structure: 512 -> 256 -> 128
        self.hidden_layer = nn.Sequential(
            # Layer 1
            nn.Linear(self.concat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.dropout_prob),
            # Layer 2
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Dropout(self.dropout_prob),
            # Layer 3
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim // 4),
            nn.Dropout(self.dropout_prob),
        )

        # 3. Output Head
        # Binary classification: Output dimension 1 with Sigmoid activation
        self.output_layer = nn.Linear(self.hidden_dim // 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous, categorical):
        """
        Forward pass of the model.

        Args:
            continuous (torch.Tensor): Continuous features of shape (Batch, 30)
            categorical (torch.Tensor): Categorical indices of shape (Batch, 10)

        Returns:
            torch.Tensor: Probability of class 1, shape (Batch, 1)
        """
        # Process Categorical: Embed and Flatten
        # (B, 10) -> (B, 10, 16)
        emb = self.embedding(categorical)
        # Flatten to (B, 160)
        emb_flat = emb.view(emb.size(0), -1)

        # Concatenate with Continuous Features
        # (B, 160) + (B, 30) -> (B, 190)
        x = torch.cat([emb_flat, continuous], dim=1)

        # Hidden Layer
        x = self.hidden_layer(x)

        # Output Layer
        logits = self.output_layer(x)
        probs = self.sigmoid(logits)

        return probs


def train_model(model, train_loader, val_loader, device=Config.DEVICE):
    """
    Trains the ShallowEmbeddingMLP model with Early Stopping.

    Args:
        model (nn.Module): The model instance.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to train on.

    Returns:
        nn.Module: The trained model with best weights loaded.
    """
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCELoss()

    # Tracking
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Unpack batch
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(cont, cat)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                cont = batch["continuous"].to(device)
                cat = batch["categorical"].to(device)
                targets = batch["target"].to(device)

                outputs = model(cont, cat)

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        # Calculate Metric
        val_auc = roc_auc_score(val_targets, val_preds)

        # Print full precision as requested
        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc}"
        )

        # --- Early Stopping Logic ---
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best AUC: {best_auc}"
                )
                break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights.")

    return model


def predict(model, test_loader, device=Config.DEVICE):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    model.to(device)
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)

            outputs = model(cont, cat)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds)


def save_submission(predictions):
    """
    Saves predictions to submission.csv using IDs from metadata.

    Args:
        predictions (np.ndarray): Predicted probabilities for the test set.
    """
    # Load test metadata to get IDs
    test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    df_test_meta = pd.read_csv(test_meta_path)
    test_ids = df_test_meta["id"].values

    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions."
        )

    # Flatten predictions if necessary (N, 1) -> (N,)
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
