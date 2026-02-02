import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import MLP_PARAMS, DEVICE, CACHE_DIR, RANDOM_STATE
from library.data_factory import DataBuilder
from library.utils import set_seed, get_device

# Ensure cache directory exists for model saving
os.makedirs(CACHE_DIR, exist_ok=True)


class DualAttentionCentroidNet(nn.Module):
    """
    Centroid-Augmented Dual-Attention MLP.

    Features:
    - Dual-Query Attention (Title->History, Body->History)
    - Global Persona Injection (Centroid)
    - Metadata-driven Credibility Gating
    - Dropout-only regularization (No BatchNorm)
    """

    def __init__(self, meta_dim, params):
        super(DualAttentionCentroidNet, self).__init__()

        self.emb_dim = params["embedding_dim"]
        self.att_dim = params["attention_dim"]
        self.hidden_dims = params["hidden_dims"]
        self.dropout_rate = params["dropout_rate"]
        self.emb_dropout_rate = params["embedding_dropout"]

        # 1. Embedding Dropout
        self.emb_dropout = nn.Dropout(self.emb_dropout_rate)

        # 2. Attention Projections
        # We project Query and Key to attention_dim for score calculation.
        # Values (History) are used in their original embedding dimension.
        self.q_title_proj = nn.Linear(self.emb_dim, self.att_dim)
        self.q_body_proj = nn.Linear(self.emb_dim, self.att_dim)
        self.k_hist_proj = nn.Linear(self.emb_dim, self.att_dim)

        # 3. Gating Mechanism
        # Calculates the size of the fusion vector to determine gate size
        # Fusion = Title + Body + Topic_Ctx + Narrative_Ctx + Centroid + Meta
        self.fusion_dim = (self.emb_dim * 5) + meta_dim

        # Gate projects metadata to fusion dimension
        self.credibility_gate = nn.Linear(meta_dim, self.fusion_dim)

        # 4. MLP Head
        layers = []
        input_dim = self.fusion_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Final Output
        self.output_layer = nn.Linear(input_dim, 1)

    def _compute_attention(self, query, key, value, mask):
        """
        Computes Dot-Product Attention.
        Query: (B, Emb)
        Key: (B, Seq, Emb)
        Value: (B, Seq, Emb)
        Mask: (B, Seq)
        """
        # Project Q and K
        # Q: (B, Att)
        q_proj = query  # Assumes query passed is already projected or we project here.
        # Actually, self.q_proj is defined in init, so we expect raw input to forward and project there.
        # Let's handle projection inside forward for clarity.

        # Calculate Scores: (B, 1, Att) @ (B, Att, Seq) -> (B, 1, Seq)
        # Note: query is (B, Att), key is (B, Seq, Att)
        scores = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))

        # Scale
        scores = scores / (self.att_dim**0.5)

        # Apply Mask
        # Mask is (B, Seq). We need (B, 1, Seq). 0 indicates padding.
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Weights
        attn_weights = F.softmax(scores, dim=-1)

        # Context: (B, 1, Seq) @ (B, Seq, Emb) -> (B, 1, Emb)
        context = torch.bmm(attn_weights, value)

        return context.squeeze(1)

    def forward(self, title_emb, body_emb, history_seq, history_mask, centroid, meta):
        # Apply Dropout to embeddings
        title_emb = self.emb_dropout(title_emb)
        body_emb = self.emb_dropout(body_emb)
        history_seq = self.emb_dropout(history_seq)
        centroid = self.emb_dropout(centroid)

        # --- Branch 3: Dual-Query Attention ---

        # Project Keys once
        # history_seq: (B, Seq, Emb) -> (B, Seq, Att)
        k_hist = self.k_hist_proj(history_seq)

        # Head A: Topic Context (Query = Title)
        q_title = self.q_title_proj(title_emb)
        topic_context = self._compute_attention(
            q_title, k_hist, history_seq, history_mask
        )

        # Head B: Narrative Context (Query = Body)
        q_body = self.q_body_proj(body_emb)
        narrative_context = self._compute_attention(
            q_body, k_hist, history_seq, history_mask
        )

        # --- Fusion & Gating ---

        # Concatenate all semantic signals + metadata
        # Dimensions: 384*5 + Meta
        fusion_vec = torch.cat(
            [title_emb, body_emb, topic_context, narrative_context, centroid, meta],
            dim=1,
        )

        # Generate Gate from Metadata
        gate = torch.sigmoid(self.credibility_gate(meta))

        # Apply Gate (Modulate fusion vector)
        gated_vec = fusion_vec * gate

        # --- MLP Head ---
        features = self.mlp(gated_vec)
        logits = self.output_layer(features)

        return logits.squeeze(1)


class MLPTrainer:
    def __init__(self, model, device, params):
        self.model = model.to(device)
        self.device = device
        self.params = params
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        self.best_auc = 0.0
        self.best_model_path = os.path.join(CACHE_DIR, "best_mlp_model.pth")

    def train(self, train_loader, val_loader):
        print(f"Starting MLP training on {self.device}...")
        patience_counter = 0

        for epoch in range(self.params["epochs"]):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # Move batch to device
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                cent = batch["centroid"].to(self.device)
                meta = batch["meta"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(title, body, hist, mask, cent, meta)
                loss = self.criterion(logits, labels)

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params["grad_clip"]
                )

                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_auc = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{self.params['epochs']} - Loss: {avg_train_loss:.6f} - Val AUC: {val_auc}"
            )

            # Early Stopping & Checkpointing
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= self.params["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if os.path.exists(self.best_model_path):
            print(f"Loading best model with AUC: {self.best_auc}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        return self.best_auc

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                cent = batch["centroid"].to(self.device)
                meta = batch["meta"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(title, body, hist, mask, cent, meta)
                probs = torch.sigmoid(logits)

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if len(all_labels) == 0:
            return 0.0

        return roc_auc_score(all_labels, all_preds)

    def predict(self, test_loader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                cent = batch["centroid"].to(self.device)
                meta = batch["meta"].to(self.device)

                logits = self.model(title, body, hist, mask, cent, meta)
                probs = torch.sigmoid(logits)

                all_preds.extend(probs.cpu().numpy())

        return np.array(all_preds)


def run_mlp_pipeline(load_cached_data=True):
    """
    Orchestrates the MLP pipeline.
    """
    set_seed(RANDOM_STATE)
    device = get_device()

    # 1. Load Data
    print("Initializing DataBuilder for MLP...")
    data_builder = DataBuilder()
    train_loader, val_loader, test_loader = data_builder.get_mlp_loaders(
        batch_size=MLP_PARAMS["batch_size"]
    )

    # 2. Determine Metadata Dimension
    # Fetch one batch to inspect shape
    sample_batch = next(iter(train_loader))
    meta_dim = sample_batch["meta"].shape[1]
    print(f"Detected Metadata Dimension: {meta_dim}")

    # 3. Initialize Model
    model = DualAttentionCentroidNet(meta_dim=meta_dim, params=MLP_PARAMS)

    # 4. Initialize Trainer
    trainer = MLPTrainer(model, device, MLP_PARAMS)

    # 5. Train
    val_auc = trainer.train(train_loader, val_loader)

    # 6. Predict on Validation (for ensemble) and Test
    print("Generating MLP predictions...")
    val_preds = trainer.predict(val_loader)  # Predict returns probabilities
    test_preds = trainer.predict(test_loader)

    return {"val_preds": val_preds, "test_preds": test_preds, "val_auc": val_auc}
