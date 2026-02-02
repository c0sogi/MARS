import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.dataset import create_dataloaders
from library.data_loader import get_processed_features
from library.utils import save_submission

# ==========================================
# Sub-Modules
# ==========================================


class DualQueryAttention(nn.Module):
    """
    Attends to user history using Title and Body as separate queries.
    This allows the model to extract different relevant historical contexts
    based on the specific semantic content of the request title versus the body.
    """

    def __init__(self, embedding_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Project history embeddings to hidden dimension
        self.history_proj = nn.Linear(embedding_dim, hidden_dim)

        # Dual Attention Heads
        self.attention_title = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.attention_body = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, title_feat, body_feat, history_emb, history_mask):
        """
        Args:
            title_feat: (B, H) Projected title features
            body_feat: (B, H) Projected body features
            history_emb: (B, Seq, E) Raw history embeddings
            history_mask: (B, Seq) 1 for valid, 0 for padding
        """
        # Project history: (B, Seq, H)
        h_feat = F.relu(self.history_proj(self.dropout(history_emb)))

        # Prepare mask for MultiheadAttention (True indicates padding)
        key_padding_mask = history_mask == 0

        # Query 1: Title -> (B, 1, H)
        q1 = title_feat.unsqueeze(1)
        attn_out1, _ = self.attention_title(
            q1, h_feat, h_feat, key_padding_mask=key_padding_mask
        )
        ctx1 = attn_out1.squeeze(1)

        # Query 2: Body -> (B, 1, H)
        q2 = body_feat.unsqueeze(1)
        attn_out2, _ = self.attention_body(
            q2, h_feat, h_feat, key_padding_mask=key_padding_mask
        )
        ctx2 = attn_out2.squeeze(1)

        return ctx1, ctx2


class GatedFusion(nn.Module):
    """
    Modulates semantic features using a clean control signal derived strictly
    from numerical metadata. This enforces orthogonal information flow, ensuring
    that the gating mechanism is not contaminated by the semantic content itself.
    """

    def __init__(self, metadata_dim, hidden_dim):
        super().__init__()
        self.gate_control = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Scalar gate
            nn.Sigmoid(),
        )

    def forward(self, semantic_vec, metadata_dense):
        """
        Args:
            semantic_vec: (B, H) Fused semantic features
            metadata_dense: (B, M) Numerical metadata features
        """
        gate = self.gate_control(metadata_dense)
        return semantic_vec * gate


# ==========================================
# Main Architecture
# ==========================================


class OrthogonalSkipGatedMLP(nn.Module):
    """
    Modular implementation of the Orthogonal Skip-Gated MLP.
    Integrates Dual-Query Attention, Global Persona Injection, and
    Orthogonal Gating with a Skip Connection.
    """

    def __init__(self, metadata_dim, skip_dim, embedding_dim=384, hidden_dim=256):
        super().__init__()

        # Feature Projections
        self.title_proj = nn.Linear(embedding_dim, hidden_dim)
        self.body_proj = nn.Linear(embedding_dim, hidden_dim)
        self.persona_proj = nn.Linear(embedding_dim, hidden_dim)

        # Sub-modules
        self.dual_attention = DualQueryAttention(
            embedding_dim, hidden_dim, dropout=Config.MLP_DROPOUT_EMB
        )

        # Semantic Fusion Layer
        # Inputs: Title + Body + Context_Title + Context_Body + Persona_Centroid
        self.semantic_fusion_dim = hidden_dim * 5
        self.semantic_fusion = nn.Linear(self.semantic_fusion_dim, hidden_dim)

        # Orthogonal Gating Mechanism
        self.gated_fusion = GatedFusion(metadata_dim, hidden_dim)

        # Final Classifier with Skip Connection
        # Input: Gated Semantic Features + Skip Features (Metadata + Community)
        self.final_input_dim = hidden_dim + skip_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(Config.MLP_DROPOUT_EMB)

    def forward(
        self,
        title_emb,
        body_emb,
        history_emb,
        history_mask,
        persona_centroid,
        metadata_dense,
        metadata_skip,
    ):
        # 1. Project Semantics
        t_feat = F.relu(self.title_proj(self.dropout(title_emb)))
        b_feat = F.relu(self.body_proj(self.dropout(body_emb)))
        p_feat = F.relu(self.persona_proj(self.dropout(persona_centroid)))

        # 2. Dual Query Attention (History Context)
        ctx1, ctx2 = self.dual_attention(t_feat, b_feat, history_emb, history_mask)

        # 3. Semantic Fusion
        semantic_raw = torch.cat([t_feat, b_feat, ctx1, ctx2, p_feat], dim=1)
        semantic_vec = F.relu(self.semantic_fusion(semantic_raw))

        # 4. Orthogonal Gating (Control Signal)
        gated_semantic = self.gated_fusion(semantic_vec, metadata_dense)

        # 5. Skip Connection & Classification
        combined = torch.cat([gated_semantic, metadata_skip], dim=1)
        logits = self.classifier(combined)

        return logits


# ==========================================
# Training & Execution Logic
# ==========================================


def train_mlp_model(
    train_loader,
    val_loader,
    input_dims,
    device,
    epochs=Config.MLP_EPOCHS,
    patience=Config.MLP_PATIENCE,
):
    """
    Trains the OrthogonalSkipGatedMLP model with Early Stopping.
    """
    model = OrthogonalSkipGatedMLP(
        metadata_dim=input_dims["metadata_dim"],
        skip_dim=input_dims["skip_dim"],
        embedding_dim=384,
        hidden_dim=Config.MLP_HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.MLP_LR, weight_decay=Config.MLP_WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    print(f"Starting MLP training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            # Move batch to device
            t = batch["title_emb"].to(device)
            b = batch["body_emb"].to(device)
            h = batch["history_emb"].to(device)
            m = batch["history_mask"].to(device)
            p = batch["persona_centroid"].to(device)
            md = batch["metadata_dense"].to(device)
            ms = batch["metadata_skip"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(t, b, h, m, p, md, ms)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                t = batch["title_emb"].to(device)
                b = batch["body_emb"].to(device)
                h = batch["history_emb"].to(device)
                m = batch["history_mask"].to(device)
                p = batch["persona_centroid"].to(device)
                md = batch["metadata_dense"].to(device)
                ms = batch["metadata_skip"].to(device)
                y = batch["label"].to(device)

                logits = model(t, b, h, m, p, md, ms)
                probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().numpy().flatten())
                val_targets.extend(y.cpu().numpy().flatten())

        val_auc = roc_auc_score(val_targets, val_preds)
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.8f} | Val AUC: {val_auc:.8f}"
        )

        # Early Stopping Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mlp(model, loader, device):
    """
    Generates predictions using the trained MLP model.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            t = batch["title_emb"].to(device)
            b = batch["body_emb"].to(device)
            h = batch["history_emb"].to(device)
            m = batch["history_mask"].to(device)
            p = batch["persona_centroid"].to(device)
            md = batch["metadata_dense"].to(device)
            ms = batch["metadata_skip"].to(device)

            logits = model(t, b, h, m, p, md, ms)
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy().flatten())
    return np.array(preds)


def run_pipeline():
    """
    Executes the full Hybrid Ensemble pipeline:
    1. Loads processed data.
    2. Trains Random Forest (Stream A).
    3. Trains MLP (Stream B).
    4. Ensembles predictions.
    5. Saves submission.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    print("Loading and processing data...")
    data = get_processed_features(load_cached_data=True)

    # 2. Train RF (Stream A)
    print("Training Random Forest (Stream A)...")
    rf = RandomForestClassifier(**Config.RF_PARAMS)
    rf.fit(data["rf_train"]["X"], data["y_train"])

    # RF Validation
    rf_val_preds = rf.predict_proba(data["rf_val"]["X"])[:, 1]
    rf_val_auc = roc_auc_score(data["y_val"], rf_val_preds)
    print(f"Random Forest Val AUC: {rf_val_auc:.8f}")

    # RF Inference
    rf_test_preds = rf.predict_proba(data["rf_test"]["X"])[:, 1]

    # 3. Train MLP (Stream B)
    print("Training MLP (Stream B)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.MLP_BATCH_SIZE
    )

    # Infer input dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    input_dims = {
        "metadata_dim": sample_batch["metadata_dense"].shape[1],
        "skip_dim": sample_batch["metadata_skip"].shape[1],
    }

    mlp_model = train_mlp_model(train_loader, val_loader, input_dims, device)
    mlp_test_preds = predict_mlp(mlp_model, test_loader, device)

    # 4. Ensemble (Simple Average)
    print("Ensembling predictions...")
    final_preds = 0.5 * rf_test_preds + 0.5 * mlp_test_preds

    # 5. Submission
    save_submission(data["ids_test"], final_preds)


# Execute the pipeline
run_pipeline()
