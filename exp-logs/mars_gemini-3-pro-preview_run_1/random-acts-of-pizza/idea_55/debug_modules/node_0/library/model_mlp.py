import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from library.config import Config
from library.utils import set_seed, get_device, calculate_roc_auc


class DualQueryAttention(nn.Module):
    """
    Computes attention context for a specific query (Title or Body) over the User History sequence.
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5
        # We use a simple dot-product attention without learnable weights for the projection
        # to keep it strictly based on semantic similarity, as per the "Dual-Query" design.
        # However, to allow the model to learn *which* parts are important, we can add
        # a linear projection for Q, K, V. Given the prompt implies "Raw SBERT",
        # we will stick to direct dot product but allow a learnable output projection if needed.
        # Here we implement standard Scaled Dot Product Attention.

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (Batch, Embed_Dim)
            key: (Batch, Seq_Len, Embed_Dim)
            value: (Batch, Seq_Len, Embed_Dim)
            mask: (Batch, Seq_Len) - 1 for valid, 0 for padding
        Returns:
            context: (Batch, Embed_Dim)
        """
        # Expand query to (Batch, 1, Embed_Dim)
        q = query.unsqueeze(1)

        # Calculate attention scores: (Batch, 1, Seq_Len)
        # key.transpose(-2, -1) -> (Batch, Embed_Dim, Seq_Len)
        scores = torch.matmul(q, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask is (Batch, Seq_Len). Unsqueeze to (Batch, 1, Seq_Len)
            mask = mask.unsqueeze(1)
            # Apply additive masking: -inf where mask is 0
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # Calculate context: (Batch, 1, Seq_Len) * (Batch, Seq_Len, Embed_Dim) -> (Batch, 1, Embed_Dim)
        context = torch.matmul(attn_weights, value)

        # Squeeze back to (Batch, Embed_Dim)
        return context.squeeze(1)


class SkipGatedMLP(nn.Module):
    """
    Orthogonal Skip-Gated Dual-Query MLP.

    Structure:
    1. Semantic Branch: Title, Body, History Attention (Topic & Narrative), Persona Centroid.
    2. Metadata Branch: Numerical features + Consistency Scalars.
    3. Gating: Metadata -> Sigmoid Gate -> Modulates Semantic Branch.
    4. Fusion: Concatenate [Gated Semantic, Raw Metadata].
    """

    def __init__(self, metadata_dim):
        super(SkipGatedMLP, self).__init__()

        self.embed_dim = Config.SBERT_EMBEDDING_DIM

        # Attention Modules
        self.topic_attention = DualQueryAttention(self.embed_dim)
        self.narrative_attention = DualQueryAttention(self.embed_dim)

        # Semantic Vector Dimension:
        # Title (384) + Body (384) + TopicContext (384) + NarrativeContext (384) + Centroid (384)
        self.semantic_dim = self.embed_dim * 5

        # Gate Generator: Metadata -> Semantic Dim
        # "Crucially, exclude embeddings to prevent gate contamination"
        self.gate_generator = nn.Sequential(
            nn.Linear(metadata_dim, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(Config.MLP_HIDDEN_DIM, self.semantic_dim),
            nn.Sigmoid(),
        )

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(Config.MLP_DROPOUT_EMB)

        # Final Classification Head
        # Input: Semantic Dim (Gated) + Metadata Dim (Skip Connection)
        fusion_dim = self.semantic_dim + metadata_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.MLP_HIDDEN_DIM // 2, 1),  # Logits for BCEWithLogitsLoss
        )

    def forward(
        self, title_emb, body_emb, history_emb, history_mask, persona_centroid, metadata
    ):
        # 1. Apply Dropout to Embeddings
        t_emb = self.emb_dropout(title_emb)
        b_emb = self.emb_dropout(body_emb)
        h_emb = self.emb_dropout(history_emb)
        p_cent = self.emb_dropout(persona_centroid)

        # 2. Dual-Query Attention
        # Topic Context: Query = Title
        topic_ctx = self.topic_attention(t_emb, h_emb, h_emb, history_mask)

        # Narrative Context: Query = Body
        narr_ctx = self.narrative_attention(b_emb, h_emb, h_emb, history_mask)

        # 3. Construct Semantic Vector S
        # Concatenate [Title, Body, Topic_Ctx, Narr_Ctx, Centroid]
        semantic_vector = torch.cat([t_emb, b_emb, topic_ctx, narr_ctx, p_cent], dim=1)

        # 4. Generate Control Gate G from Metadata
        gate = self.gate_generator(metadata)

        # 5. Orthogonal Gating
        gated_semantic = semantic_vector * gate

        # 6. Skip-Connection Fusion
        # Concatenate [Gated Semantic, Metadata]
        fused_vector = torch.cat([gated_semantic, metadata], dim=1)

        # 7. Classification
        logits = self.classifier(fused_vector)
        return logits


class MLPTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )

    def fit(self, train_loader, val_loader, epochs, patience):
        best_auc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting MLP training on {self.device}...")

        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # Unpack batch
                # Order depends on how TensorDataset was created
                (t_emb, b_emb, h_emb, h_mask, p_cent, meta, y) = [
                    b.to(self.device) for b in batch
                ]

                self.optimizer.zero_grad()

                logits = self.model(t_emb, b_emb, h_emb, h_mask, p_cent, meta)
                loss = self.criterion(logits.view(-1), y.float())

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * t_emb.size(0)

            epoch_loss = train_loss / len(train_loader.dataset)

            # --- Validation Phase ---
            val_auc, val_loss = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc}"
            )

            # --- Early Stopping ---
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Val AUC: {best_auc}")

        # Load best weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        running_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                (t_emb, b_emb, h_emb, h_mask, p_cent, meta, y) = [
                    b.to(self.device) for b in batch
                ]

                logits = self.model(t_emb, b_emb, h_emb, h_mask, p_cent, meta)
                loss = self.criterion(logits.view(-1), y.float())
                running_loss += loss.item() * t_emb.size(0)

                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y.cpu().numpy())

        avg_loss = running_loss / len(loader.dataset)
        auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))
        return auc, avg_loss

    def predict_proba(self, loader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                # Test loader might not have targets, but our TensorDataset structure usually keeps it consistent.
                # If test data has no targets, we just unpack features.
                if len(batch) == 7:  # Has targets
                    (t_emb, b_emb, h_emb, h_mask, p_cent, meta, _) = [
                        b.to(self.device) for b in batch
                    ]
                else:
                    (t_emb, b_emb, h_emb, h_mask, p_cent, meta) = [
                        b.to(self.device) for b in batch
                    ]

                logits = self.model(t_emb, b_emb, h_emb, h_mask, p_cent, meta)
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                all_preds.extend(probs)

        return np.array(all_preds)


def create_dataloader(X_dict, y=None, batch_size=32, shuffle=False):
    """
    Helper to create a DataLoader from the dictionary of tensors.
    """
    # Extract tensors from dict
    t_emb = X_dict["title_emb"]
    b_emb = X_dict["body_emb"]
    h_emb = X_dict["history_emb"]
    h_mask = X_dict["history_mask"]
    p_cent = X_dict["persona_centroid"]
    meta = X_dict["metadata"]

    # Ensure all are float tensors (except mask maybe, but float is fine for matmul)
    t_emb = t_emb.float()
    b_emb = b_emb.float()
    h_emb = h_emb.float()
    h_mask = h_mask.float()
    p_cent = p_cent.float()
    meta = meta.float()

    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32)  # BCE expects float target
        dataset = TensorDataset(t_emb, b_emb, h_emb, h_mask, p_cent, meta, y_tensor)
    else:
        dataset = TensorDataset(t_emb, b_emb, h_emb, h_mask, p_cent, meta)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_mlp_model(X_train_dict, y_train, X_val_dict, y_val):
    """
    Orchestrates MLP training.
    """
    set_seed(Config.SEED)
    device = get_device()

    # Create DataLoaders
    train_loader = create_dataloader(
        X_train_dict, y_train, Config.MLP_BATCH_SIZE, shuffle=True
    )
    val_loader = create_dataloader(
        X_val_dict, y_val, Config.MLP_BATCH_SIZE, shuffle=False
    )

    # Determine metadata dimension
    meta_dim = X_train_dict["metadata"].shape[1]

    # Initialize Model
    model = SkipGatedMLP(metadata_dim=meta_dim)

    # Initialize Trainer
    trainer = MLPTrainer(model, device)

    # Train
    trained_model = trainer.fit(
        train_loader, val_loader, epochs=Config.MLP_EPOCHS, patience=Config.MLP_PATIENCE
    )

    return trained_model, trainer


def predict_mlp(model, X_test_dict):
    """
    Generate predictions using the trained model.
    """
    set_seed(Config.SEED)
    device = get_device()

    # Create DataLoader (no shuffle, no targets)
    # Note: X_test_dict keys must match what create_dataloader expects
    test_loader = create_dataloader(
        X_test_dict, y=None, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
    )

    trainer = MLPTrainer(model, device)  # Re-use trainer wrapper for inference logic
    probs = trainer.predict_proba(test_loader)

    return probs
