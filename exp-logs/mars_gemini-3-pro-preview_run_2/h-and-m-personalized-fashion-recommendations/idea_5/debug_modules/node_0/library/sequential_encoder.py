import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time
from pathlib import Path
from library import config
from library import utils

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation Model.
    Uses a Transformer Encoder with Causal Masking to learn user state representations.
    """

    def __init__(self, vocab_size, params):
        super(SASRec, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = params["max_seq_len"]
        self.embedding_dim = params["embedding_dim"]
        self.n_layers = params["n_layers"]
        self.n_heads = params["n_heads"]
        self.dropout_rate = params["dropout"]
        self.device = config.DEVICE

        # Embeddings
        self.item_embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        # Transformer Encoder
        # We use standard PyTorch TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=self.dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq):
        # item_seq: (Batch, Seq_Len)
        seq_len = item_seq.size(1)

        # Create Position Indices: [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand(item_seq.size(0), -1)

        # Embeddings
        seq_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(self.layer_norm(seq_emb + pos_emb))

        # Causal Mask (Look-ahead mask)
        # Prevents attending to future positions
        # Mask shape: (Seq_Len, Seq_Len)
        # 0 (False) = attend, -inf (True) = mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device), diagonal=1
        ).bool()

        # Transformer Pass
        # output: (Batch, Seq_Len, Embed_Dim)
        output = self.transformer_encoder(x, mask=mask, is_causal=True)

        return output

    def predict_logits(self, item_seq):
        """
        Returns logits for all items in vocabulary for each step in sequence.
        Used during training.
        """
        # (Batch, Seq_Len, Embed_Dim)
        output = self.forward(item_seq)
        # (Batch, Seq_Len, Vocab_Size)
        logits = torch.matmul(output, self.item_embedding.weight.transpose(0, 1))
        return logits


# ==========================================
# 2. DATASET
# ==========================================


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: Tensor of shape (N_users, Max_Len)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Input: [x1, x2, ..., x_{T-1}]
        # Target: [x2, x3, ..., x_T]
        # We assume sequence is padded at the beginning if short,
        # or full length.

        # We strip the last item for input, and the first item for target
        # This creates the shift.
        input_seq = seq[:-1]
        target_seq = seq[1:]

        return input_seq, target_seq


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================


def train_sequential_model(data_dict, params=None, load_cached_data=False):
    """
    Trains the SASRec model or loads it from cache.

    Args:
        data_dict: Output from data_loader.preprocess_sequences
        params: Hyperparameters dict (defaults to config.SEQ_CONFIG)
        load_cached_data: Whether to load pre-trained model

    Returns:
        Trained SASRec model
    """
    if params is None:
        params = config.SEQ_CONFIG

    model_path = config.SEQ_MODEL_PATH

    # 1. Check Cache
    if load_cached_data and model_path.exists():
        print(f"Loading cached Sequential Model from {model_path}...")
        try:
            # We need to instantiate the model structure first
            vocab_size = data_dict["vocab_size"]
            model = SASRec(vocab_size, params)
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.to(config.DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model cache: {e}. Retraining...")

    # 2. Setup Data
    print("Setting up Sequential Model training...")
    sequences = data_dict["sequences"]
    vocab_size = data_dict["vocab_size"]

    # Split into Train/Val (internal split for monitoring)
    # We use 90/10 split
    n_total = len(sequences)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val

    # Use generator for reproducibility
    generator = torch.Generator().manual_seed(config.RANDOM_STATE)
    train_subset, val_subset = random_split(
        SequenceDataset(sequences), [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Setup Model
    utils.seed_everything(config.RANDOM_STATE)
    model = SASRec(vocab_size, params).to(config.DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    # Ignore padding index 0 in loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {config.DEVICE}...")
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    for epoch in range(params["epochs"]):
        start_time = time.time()

        # --- Training ---
        model.train()
        total_train_loss = 0.0

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq = input_seq.to(config.DEVICE)
            target_seq = target_seq.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward
            logits = model.predict_logits(input_seq)

            # Flatten for loss: (Batch * Seq, Vocab) vs (Batch * Seq)
            logits = logits.view(-1, vocab_size)
            targets = target_seq.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for input_seq, target_seq in enumerate(val_loader):
                # Unpack tuple correctly
                input_seq, target_seq = target_seq  # DataLoader yields (input, target)

                input_seq = input_seq.to(config.DEVICE)
                target_seq = target_seq.to(config.DEVICE)

                logits = model.predict_logits(input_seq)
                logits = logits.view(-1, vocab_size)
                targets = target_seq.view(-1)

                loss = criterion(logits, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{params['epochs']} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {avg_train_loss} | "
            f"Val Loss: {avg_val_loss}"
        )

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= params["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model state
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    return model


# ==========================================
# 4. EMBEDDING EXTRACTION
# ==========================================


def extract_embeddings(model, data_dict, batch_size=1024):
    """
    Extracts user state vectors and item embedding matrix from the trained model.

    Returns:
        user_embeddings (np.ndarray): (N_users, Embed_Dim)
        item_embeddings (np.ndarray): (Vocab_Size, Embed_Dim)
    """
    print("Extracting embeddings from Sequential Model...")

    model.eval()
    sequences = data_dict["sequences"]

    user_embeddings_list = []

    # Create a simple loader for inference
    loader = DataLoader(
        sequences, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS
    )

    with torch.no_grad():
        for batch_seq in loader:
            batch_seq = batch_seq.to(config.DEVICE)

            # Forward pass to get hidden states
            # Output: (Batch, Seq_Len, Embed_Dim)
            output = model(batch_seq)

            # We want the state corresponding to the last item in the sequence
            # Since sequences are fixed length (padded at start), the last item is always at index -1
            # Note: If we used right-padding, we'd need to gather based on lengths.
            # With left-padding [0, 0, A, B], the last state corresponds to B, which is correct.
            last_state = output[:, -1, :]

            user_embeddings_list.append(last_state.cpu().numpy())

    user_embeddings = np.concatenate(user_embeddings_list, axis=0)

    # Extract Item Embeddings directly from the layer
    item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

    print(f"Extracted User Embeddings: {user_embeddings.shape}")
    print(f"Extracted Item Embeddings: {item_embeddings.shape}")

    return user_embeddings, item_embeddings
