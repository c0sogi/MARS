import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.utils import set_seed, compute_auc


class PizzaDataset(Dataset):
    """
    Custom Dataset to handle multi-modal inputs:
    - text: Request SBERT embedding
    - hist: Sequence of Subreddit SBERT embeddings
    - mask: Attention mask for history sequence
    - meta: Engineered metadata features
    - y: Target labels (optional)
    """

    def __init__(self, text, hist, mask, meta, y=None):
        self.text = torch.FloatTensor(text)
        self.hist = torch.FloatTensor(hist)
        self.mask = torch.FloatTensor(mask)
        self.meta = torch.FloatTensor(meta)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sample = {
            "text": self.text[idx],
            "hist": self.hist[idx],
            "mask": self.mask[idx],
            "meta": self.meta[idx],
        }
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample


class AttentionGatedNet(nn.Module):
    """
    Neural Network with Attention-Gated Fusion.

    1. Attends to user history based on request semantics.
    2. Generates a 'Credibility Gate' from metadata.
    3. Fuses semantic signals with the credibility gate.
    """

    def __init__(self, input_dim_text, input_dim_meta, hidden_dim, dropout_rate):
        super(AttentionGatedNet, self).__init__()

        # --- Attention Mechanism ---
        # Scale factor for dot-product attention
        self.scale = 1.0 / (input_dim_text**0.5)

        # Semantic Dimension = Request (384) + Context (384) = 768
        self.semantic_dim = input_dim_text * 2

        # --- Metadata Gate ---
        # Projects metadata to a gate vector matching the semantic dimension
        self.meta_gate_net = nn.Sequential(
            nn.Linear(input_dim_meta, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.semantic_dim),
            nn.Sigmoid(),  # Sigmoid to act as a gate (0 to 1)
        )

        # --- Final Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, hist, mask, meta):
        # text: (Batch, Emb_Dim)
        # hist: (Batch, Seq_Len, Emb_Dim)
        # mask: (Batch, Seq_Len)
        # meta: (Batch, Meta_Dim)

        # 1. Attention Branch
        # Query: Request Text -> (Batch, 1, Emb_Dim)
        query = text.unsqueeze(1)

        # Keys: History -> (Batch, Seq_Len, Emb_Dim)
        # Scores: (Batch, 1, Seq_Len)
        scores = torch.bmm(query, hist.transpose(1, 2)) * self.scale

        # Apply Masking (set padded positions to very small value)
        # mask is 1.0 for valid, 0.0 for padding.
        # We check where mask == 0 and fill with -1e9
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # Attention Weights
        attn_weights = F.softmax(scores, dim=-1)  # (Batch, 1, Seq_Len)

        # Context Vector: Weighted sum of history
        context = torch.bmm(attn_weights, hist).squeeze(1)  # (Batch, Emb_Dim)

        # Combined Semantic Feature
        semantic_feat = torch.cat([text, context], dim=1)  # (Batch, 2*Emb_Dim)
        semantic_feat = self.dropout(semantic_feat)

        # 2. Gating Branch
        gate = self.meta_gate_net(meta)  # (Batch, 2*Emb_Dim)

        # 3. Fusion
        # Modulate semantics by credibility gate
        gated_feat = semantic_feat * gate

        # 4. Classification
        logits = self.classifier(gated_feat)
        return logits


def train_mlp_model(mlp_features):
    """
    Trains the AttentionGatedNet using features provided in the dictionary.

    Args:
        mlp_features (dict): Dictionary containing numpy arrays for train/val splits.
                             Keys: text_train, hist_train, mask_train, meta_train, y_train, etc.

    Returns:
        model: Trained PyTorch model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on device: {device}")

    # --- Prepare Data ---
    train_dataset = PizzaDataset(
        mlp_features["text_train"],
        mlp_features["hist_train"],
        mlp_features["mask_train"],
        mlp_features["meta_train"],
        mlp_features["y_train"],
    )

    val_dataset = PizzaDataset(
        mlp_features["text_val"],
        mlp_features["hist_val"],
        mlp_features["mask_val"],
        mlp_features["meta_val"],
        mlp_features["y_val"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
    )

    # --- Initialize Model ---
    input_dim_text = mlp_features["text_train"].shape[1]
    input_dim_meta = mlp_features["meta_train"].shape[1]

    model = AttentionGatedNet(
        input_dim_text=input_dim_text,
        input_dim_meta=input_dim_meta,
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout_rate=Config.MLP_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.MLP_LR, weight_decay=Config.MLP_WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(
        f"Starting MLP training for {Config.MLP_EPOCHS} epochs with patience {Config.MLP_PATIENCE}..."
    )

    for epoch in range(Config.MLP_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            text = batch["text"].to(device)
            hist = batch["hist"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            y = batch["y"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(text, hist, mask, meta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * text.size(0)

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                text = batch["text"].to(device)
                hist = batch["hist"].to(device)
                mask = batch["mask"].to(device)
                meta = batch["meta"].to(device)
                y = batch["y"].to(device).unsqueeze(1)

                logits = model(text, hist, mask, meta)
                loss = criterion(logits, y)
                val_loss += loss.item() * text.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = compute_auc(val_targets, val_preds)

        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f} - Val AUC: {val_auc:.10f}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.MLP_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc:.10f}"
            )
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model weights.")

    return model


def predict_mlp_model(model, mlp_features, split="test"):
    """
    Generates predictions for the specified split.
    """
    device = next(model.parameters()).device
    model.eval()

    dataset = PizzaDataset(
        mlp_features[f"text_{split}"],
        mlp_features[f"hist_{split}"],
        mlp_features[f"mask_{split}"],
        mlp_features[f"meta_{split}"],
    )

    loader = DataLoader(dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False)

    preds = []
    print(f"Generating MLP predictions for {len(dataset)} {split} samples...")

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            hist = batch["hist"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"].to(device)

            logits = model(text, hist, mask, meta)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)

    return np.array(preds).flatten()
