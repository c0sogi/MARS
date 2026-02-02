import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
from library.config import (
    EMBEDDING_DIM,
    MLP_HIDDEN_DIM,
    MLP_DROPOUT_EMB,
    MLP_DROPOUT_DENSE,
    MLP_LEARNING_RATE,
    MLP_WEIGHT_DECAY,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_PATIENCE,
    DEVICE,
)


class DualQueryAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention where the Query comes from one source (e.g., Title)
    and Key/Value come from a sequence (e.g., User History).
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        # Linear projections to map inputs to a common attention space
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (Batch, Dim)
            key: (Batch, Seq_Len, Dim)
            value: (Batch, Seq_Len, Dim)
            mask: (Batch, Seq_Len) - 1 for valid, 0 for padding
        Returns:
            context: (Batch, Dim)
        """
        # Project and reshape Query to (Batch, 1, Dim)
        Q = self.q_proj(query).unsqueeze(1)

        # Project Key and Value
        K = self.k_proj(key)  # (Batch, Seq_Len, Dim)
        V = self.v_proj(value)  # (Batch, Seq_Len, Dim)

        # Compute Attention Scores: (Batch, 1, Dim) @ (Batch, Dim, Seq_Len) -> (Batch, 1, Seq_Len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Expand mask to (Batch, 1, Seq_Len)
            mask_expanded = mask.unsqueeze(1)
            # Apply additive masking: set padding positions to negative infinity
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        # Compute Attention Weights
        attn_weights = F.softmax(scores, dim=-1)

        # Compute Context: (Batch, 1, Seq_Len) @ (Batch, Seq_Len, Dim) -> (Batch, 1, Dim)
        context = torch.matmul(attn_weights, V)

        # Squeeze to (Batch, Dim)
        return context.squeeze(1)


class OrthogonalSkipGatedMLP(nn.Module):
    """
    Hybrid Neural Network with Orthogonal Skip-Gated Fusion.

    Features:
    - Dual-Query Attention for History Context (Topic & Narrative).
    - Explicit Gate derived ONLY from Metadata to modulate Semantic features.
    - Skip connection for Metadata to preserve direct signal.
    """

    def __init__(self, metadata_dim):
        super(OrthogonalSkipGatedMLP, self).__init__()

        self.embedding_dim = EMBEDDING_DIM
        self.metadata_dim = metadata_dim

        # --- Embeddings Dropout ---
        self.emb_dropout = nn.Dropout(MLP_DROPOUT_EMB)

        # --- Attention Mechanisms ---
        self.topic_attention = DualQueryAttention(EMBEDDING_DIM)
        self.narrative_attention = DualQueryAttention(EMBEDDING_DIM)

        # --- Semantic Vector Construction ---
        # Components: Title, Body, Topic_Ctx, Narr_Ctx, Centroid
        # Dimension: 384 * 5
        self.semantic_dim = EMBEDDING_DIM * 5

        # --- Orthogonal Gate Generator ---
        # Maps Metadata -> Semantic Dimension to create a filter
        self.gate_generator = nn.Sequential(
            nn.Linear(metadata_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(MLP_HIDDEN_DIM, self.semantic_dim),
            nn.Sigmoid(),  # Output range [0, 1] for gating
        )

        # --- Classifier ---
        # Input: [Gated_Semantic, Metadata]
        self.classifier_input_dim = self.semantic_dim + metadata_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(MLP_HIDDEN_DIM // 2, 1),
        )

    def forward(
        self, title_emb, body_emb, hist_seq, hist_mask, hist_centroid, metadata
    ):
        # 1. Regularize Embeddings
        title = self.emb_dropout(title_emb)
        body = self.emb_dropout(body_emb)
        hist_seq = self.emb_dropout(hist_seq)
        centroid = self.emb_dropout(hist_centroid)

        # 2. Compute Contexts via Dual-Query Attention
        # Topic Context: Title attends to History
        topic_ctx = self.topic_attention(title, hist_seq, hist_seq, hist_mask)
        # Narrative Context: Body attends to History
        narr_ctx = self.narrative_attention(body, hist_seq, hist_seq, hist_mask)

        # 3. Construct Semantic Vector S
        semantic_vector = torch.cat([title, body, topic_ctx, narr_ctx, centroid], dim=1)

        # 4. Generate Orthogonal Control Gate G
        # Depends ONLY on metadata
        gate = self.gate_generator(metadata)

        # 5. Apply Gating (Modulation)
        gated_semantic = semantic_vector * gate

        # 6. Skip-Connection Fusion
        # Concatenate gated semantics with original metadata
        fused_vector = torch.cat([gated_semantic, metadata], dim=1)

        # 7. Classification
        logits = self.classifier(fused_vector)

        return logits


def train_mlp(train_dataset, val_dataset, metadata_dim, save_path=None):
    """
    Trains the OrthogonalSkipGatedMLP model with Early Stopping.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        metadata_dim (int): Dimension of the metadata feature vector.
        save_path (str, optional): Path to save the best model weights.

    Returns:
        model: The trained model with best weights loaded.
    """
    # Initialize DataLoaders
    # num_workers=0 to ensure compatibility in restricted environments
    train_loader = DataLoader(
        train_dataset,
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Initialize Model
    model = OrthogonalSkipGatedMLP(metadata_dim).to(DEVICE)

    # Setup Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping State
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting MLP training on {DEVICE}...")

    for epoch in range(MLP_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            # Move batch to device
            title = batch["title_emb"].to(DEVICE)
            body = batch["body_emb"].to(DEVICE)
            hist_seq = batch["hist_seq"].to(DEVICE)
            hist_mask = batch["hist_mask"].to(DEVICE)
            centroid = batch["hist_centroid"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            logits = model(title, body, hist_seq, hist_mask, centroid, meta)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * title.size(0)

        avg_train_loss = train_loss_sum / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                title = batch["title_emb"].to(DEVICE)
                body = batch["body_emb"].to(DEVICE)
                hist_seq = batch["hist_seq"].to(DEVICE)
                hist_mask = batch["hist_mask"].to(DEVICE)
                centroid = batch["hist_centroid"].to(DEVICE)
                meta = batch["metadata"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                logits = model(title, body, hist_seq, hist_mask, centroid, meta)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * title.size(0)

        avg_val_loss = val_loss_sum / len(val_dataset)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{MLP_EPOCHS} - Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}"
        )

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        print(f"Restoring best model with Val Loss: {best_val_loss}")
        model.load_state_dict(best_model_state)

    return model
