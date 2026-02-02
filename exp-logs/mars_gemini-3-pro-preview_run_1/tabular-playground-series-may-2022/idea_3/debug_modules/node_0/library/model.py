import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import get_dataloaders
from library.utils import save_submission


class HybridTransformerModel(nn.Module):
    """
    Hybrid architecture combining a Transformer Encoder for sequence features
    and an MLP for numerical features.
    """

    def __init__(self, num_numerical_features):
        super().__init__()

        # ==========================================
        # Sequence Branch
        # ==========================================
        self.embedding = nn.Embedding(
            num_embeddings=Config.VOCAB_SIZE,
            embedding_dim=Config.EMBED_DIM,
            padding_idx=0,
        )

        # Learnable positional encoding: (1, Max Len, Embed Dim)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, Config.MAX_SEQ_LEN, Config.EMBED_DIM)
        )
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.TRANSFORMER_DROPOUT,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.N_TRANSFORMER_LAYERS
        )

        # ==========================================
        # Numerical Branch
        # ==========================================
        # Project numerical features to match embedding dimension
        self.num_proj = nn.Sequential(
            nn.Linear(num_numerical_features, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT),
        )

        # ==========================================
        # Fusion & Classification Head
        # ==========================================
        # Input is concatenation of Pooled Seq (Embed Dim) + Projected Num (Embed Dim)
        fusion_input_dim = Config.EMBED_DIM * 2

        mlp_layers = []
        in_dim = fusion_input_dim

        for hidden_dim in Config.MLP_HIDDEN_SIZES:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(Config.MLP_DROPOUT))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Final Output Head (Linear -> Logits)
        self.head = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for Linear and Embedding layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, sequence, numerical):
        """
        Args:
            sequence: (Batch, Seq_Len) LongTensor
            numerical: (Batch, Num_Features) FloatTensor
        Returns:
            logits: (Batch, 1) FloatTensor
        """
        # --- Sequence Branch ---
        # Create padding mask (True where padding/0) for Transformer
        # src_key_padding_mask shape: (Batch, Seq_Len)
        padding_mask = sequence == 0

        # Embed and add position
        x_seq = self.embedding(sequence)  # (Batch, Seq_Len, Embed_Dim)
        x_seq = x_seq + self.pos_encoder  # Broadcasting

        # Transformer Encoder
        x_seq = self.transformer_encoder(x_seq, src_key_padding_mask=padding_mask)

        # Global Average Pooling (ignoring padding)
        # Create a mask that is 1.0 for valid tokens and 0.0 for padding
        # (~padding_mask) is True for valid tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (Batch, Seq_Len, 1)

        sum_embeddings = torch.sum(x_seq * mask_expanded, dim=1)  # (Batch, Embed_Dim)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (Batch, 1)

        seq_pooled = sum_embeddings / sum_mask  # (Batch, Embed_Dim)

        # --- Numerical Branch ---
        num_proj = self.num_proj(numerical)  # (Batch, Embed_Dim)

        # --- Fusion ---
        combined = torch.cat([seq_pooled, num_proj], dim=1)  # (Batch, 2 * Embed_Dim)

        # --- MLP ---
        features = self.mlp(combined)
        logits = self.head(features)

        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        seq = batch["sequence"].to(device)
        num = batch["numerical"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)
        batch_size = seq.size(0)

        optimizer.zero_grad()
        logits = model(seq, num)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["sequence"].to(device)
            num = batch["numerical"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            batch_size = seq.size(0)

            logits = model(seq, num)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_loss = running_loss / dataset_size
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            num = batch["numerical"].to(device)
            ids = batch["id"].numpy()

            logits = model(seq, num)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_ids.extend(ids)

    # Save submission
    save_submission(all_ids, all_preds)


def train_model(debug=False):
    """
    Main training loop.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Determine number of numerical features dynamically
    # Check the first batch or the dataset
    sample_batch = next(iter(train_loader))
    num_numerical_features = sample_batch["numerical"].shape[1]
    print(f"Detected {num_numerical_features} numerical features.")

    # 3. Model Initialization
    model = HybridTransformerModel(num_numerical_features=num_numerical_features)
    model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.SCHEDULER_WARMUP_PCT,
        anneal_strategy="cos",
    )

    # 5. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 6. Load Best Model and Predict
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    predict_and_submit(model, test_loader, device)

    return model


def run(debug=False):
    """
    Entry point to run the full pipeline.
    """
    train_model(debug=debug)
