import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_processor import DataProcessor
from library.dataset import ManufacturingDataset

# -------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------


class NumericalFeatureTokenizer(nn.Module):
    """
    Projects each numerical feature into a distinct dense token.
    Implements: Token = (Scalar * Weight) + FeatureIdentityEmbedding
    """

    def __init__(self, num_features, d_model):
        super().__init__()
        # Weight vector for each feature: (Num_Features, d_model)
        # Initialized with scaled normal distribution for stability
        self.weights = nn.Parameter(
            torch.randn(num_features, d_model) / math.sqrt(d_model)
        )

        # Feature Identity Embedding (Bias): (Num_Features, d_model)
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x):
        # x shape: (Batch, Num_Features)
        # Expand to (Batch, Num_Features, 1) for broadcasting
        x_expanded = x.unsqueeze(-1)

        # Element-wise multiplication broadcasts over d_model dimension
        # (Batch, Num_Features, 1) * (Num_Features, d_model) -> (Batch, Num_Features, d_model)
        tokens = x_expanded * self.weights

        # Add feature identity embeddings
        tokens = tokens + self.biases
        return tokens


class SequenceEmbedder(nn.Module):
    """
    Embeds categorical sequence data and adds positional encodings.
    """

    def __init__(self, vocab_size, seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Learnable positional encoding: (1, Seq_Len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        # Embed -> (Batch, Seq_Len, d_model)
        x_emb = self.embedding(x)

        # Add positional encodings (broadcasting over batch)
        return x_emb + self.pos_embedding


class GUTClassifier(nn.Module):
    """
    Granular Unified Transformer (GUT) Classifier.
    Combines numerical tokens and sequence tokens into a single transformer stream.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. Input Processing
        self.num_tokenizer = NumericalFeatureTokenizer(
            num_features=len(config.numerical_features), d_model=config.d_model
        )

        self.seq_embedder = SequenceEmbedder(
            vocab_size=config.vocab_size,
            seq_len=config.sequence_len,
            d_model=config.d_model,
        )

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # 3. MLP Head
        layers = []
        in_dim = config.d_model
        for hidden_dim in config.mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Final projection to scalar logit
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters (Xavier Uniform for Linear/Projections)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_num, x_seq):
        # 1. Tokenize Inputs
        num_tokens = self.num_tokenizer(x_num)  # (Batch, Num_Feats, d_model)
        seq_tokens = self.seq_embedder(x_seq)  # (Batch, Seq_Len, d_model)

        # 2. Concatenate into Unified Sequence
        # Shape: (Batch, Num_Feats + Seq_Len, d_model)
        tokens = torch.cat([num_tokens, seq_tokens], dim=1)

        # 3. Apply Transformer
        context = self.transformer(tokens)

        # 4. Global Average Pooling
        # Shape: (Batch, d_model)
        pooled = context.mean(dim=1)

        # 5. MLP Head
        # Shape: (Batch, 1)
        logits = self.mlp_head(pooled)
        return logits


# -------------------------------------------------------------------
# Training Utilities
# -------------------------------------------------------------------


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_seq = batch["x_seq"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_num, x_seq)
        loss = criterion(logits, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)
            target = batch["target"].to(device)

            logits = model(x_num, x_seq)
            loss = criterion(logits, target)

            total_loss += loss.item()

            # Sigmoid for probabilities
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc = roc_auc_score(all_targets, all_preds)
    return total_loss / len(loader), auc


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------


def run_pipeline():
    # 1. Setup
    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading & Processing
    print("Processing data...")
    processor = DataProcessor(config)
    # Load cached data or process from scratch
    data = processor.process_data(load_cached_data=True)

    # 3. Dataset & Dataloader Creation
    print("Creating datasets...")
    train_dataset = ManufacturingDataset(
        data["X_num_train"], data["X_seq_train"], data["y_train"]
    )
    val_dataset = ManufacturingDataset(
        data["X_num_val"], data["X_seq_val"], data["y_val"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing GUT Classifier...")
    model = GUTClassifier(config).to(device)

    # 5. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = config.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )

    # 6. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc + config.early_stopping_min_delta:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc:.6f}")

    # 7. Inference & Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_dataset = ManufacturingDataset(data["X_num_test"], data["X_seq_test"], None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print("Generating predictions...")
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)
            logits = model(x_num, x_seq)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Save submission
    ids = data["ids_test"]
    submission = pd.DataFrame({config.id_col: ids, config.target_col: all_preds})

    print(f"Saving submission to {config.submission_path}...")
    submission.to_csv(config.submission_path, index=False)
    print("Submission saved successfully.")


# Execute pipeline
run_pipeline()
