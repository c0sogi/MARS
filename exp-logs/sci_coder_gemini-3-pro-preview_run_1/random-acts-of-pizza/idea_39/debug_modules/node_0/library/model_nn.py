import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy

from library.config import (
    DEVICE,
    SEED,
    MLP_HIDDEN_DIM,
    MLP_PROJECTION_DIM,
    MLP_DROPOUT_EMB,
    MLP_DROPOUT_DENSE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    CACHE_DIR,
)
from library.utils import set_seed, compute_auc
from library.dataset import get_dataloaders


class DualQueryAttention(nn.Module):
    """
    Computes attention between a query vector and a sequence of history vectors.
    Returns the context vector and the cosine similarity scalar.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.scale = input_dim**-0.5

    def forward(self, query, history, mask):
        # query: (B, D)
        # history: (B, S, D)
        # mask: (B, S) - 1 for valid, 0 for padding

        # Expand query for batch matrix multiplication: (B, 1, D)
        q = query.unsqueeze(1)

        # Calculate attention scores: (B, 1, D) @ (B, D, S) -> (B, 1, S)
        scores = torch.bmm(q, history.transpose(1, 2)) * self.scale

        # Apply mask: fill padding positions with -1e9
        # mask shape needs to be (B, 1, S)
        mask_expanded = mask.unsqueeze(1)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Softmax to get weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, S)

        # Compute context vector: (B, 1, S) @ (B, S, D) -> (B, 1, D)
        context = torch.bmm(attn_weights, history)
        context = context.squeeze(1)  # (B, D)

        # Compute Cosine Similarity between Query and Context
        # Handle cases where vectors might be zero (e.g. no history)
        eps = 1e-8
        q_norm = query / (query.norm(dim=1, keepdim=True) + eps)
        c_norm = context / (context.norm(dim=1, keepdim=True) + eps)

        # Dot product of normalized vectors
        similarity = (q_norm * c_norm).sum(dim=1, keepdim=True)  # (B, 1)

        return context, similarity


class CommunityAwareDualQueryMLP(nn.Module):
    """
    Hybrid Neural Network with Dual-Query Attention and Community-Aware Gating.
    """

    def __init__(self, dense_input_dim, embedding_dim=384):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Branch 3: Dual-Query History Attention
        self.attention = DualQueryAttention(embedding_dim)

        # Semantic Fusion Dimension
        # Components: Title(384) + Body(384) + Ctx_Title(384) + Ctx_Body(384) + Sim_Title(1) + Sim_Body(1)
        self.semantic_dim = (embedding_dim * 4) + 2

        # Branch 4: Credibility Gate
        # Projects metadata/TopK to a gate that matches the semantic dimension
        self.gate_mlp = nn.Sequential(
            nn.Linear(dense_input_dim, MLP_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(MLP_PROJECTION_DIM, self.semantic_dim),
            nn.Sigmoid(),
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(MLP_DROPOUT_EMB),
            nn.Linear(self.semantic_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(MLP_HIDDEN_DIM, 1),
        )

    def forward(self, title_emb, body_emb, history_seq, history_mask, dense_features):
        # Branch 1 & 2 are just the raw embeddings passed in

        # Branch 3: Attention Mechanisms
        ctx_title, sim_title = self.attention(title_emb, history_seq, history_mask)
        ctx_body, sim_body = self.attention(body_emb, history_seq, history_mask)

        # Construct Semantic Vector
        semantic_vector = torch.cat(
            [title_emb, body_emb, ctx_title, ctx_body, sim_title, sim_body], dim=1
        )

        # Branch 4: Generate Gate
        gate = self.gate_mlp(dense_features)

        # Gated Fusion
        gated_vector = semantic_vector * gate

        # Classification
        logits = self.classifier(gated_vector)

        return logits.squeeze(1)


class NeuralNetworkModel:
    """
    Wrapper for training and inference of the MLP model.
    """

    def __init__(self):
        self.device = torch.device(DEVICE)
        set_seed(SEED)

    def run(self, load_cached_data=True):
        """
        Orchestrates data loading, training, and inference.
        """
        # 1. Load Data
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=load_cached_data, batch_size=BATCH_SIZE, verbose=False
        )

        # Determine dimensions dynamically
        sample_batch = next(iter(train_loader))
        dense_dim = sample_batch["dense_features"].shape[1]
        emb_dim = sample_batch["title_emb"].shape[1]

        # 2. Initialize Model
        model = CommunityAwareDualQueryMLP(
            dense_input_dim=dense_dim, embedding_dim=emb_dim
        )
        model = model.to(self.device)

        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        # 3. Training Loop
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0

        print(f"Starting MLP Training (Epochs: {EPOCHS}, Patience: {PATIENCE})...")

        for epoch in range(EPOCHS):
            # Train
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()

                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                dense = batch["dense_features"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = model(title, body, hist, mask, dense)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    title = batch["title_emb"].to(self.device)
                    body = batch["body_emb"].to(self.device)
                    hist = batch["history_seq"].to(self.device)
                    mask = batch["history_mask"].to(self.device)
                    dense = batch["dense_features"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(title, body, hist, mask, dense)
                    probs = torch.sigmoid(logits)

                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_auc = compute_auc(val_targets, val_preds)

            # Check Early Stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"MLP Validation AUC: {best_val_auc}")

        # 4. Inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                dense = batch["dense_features"].to(self.device)

                logits = model(title, body, hist, mask, dense)
                probs = torch.sigmoid(logits)
                test_preds.extend(probs.cpu().numpy())

        # Retrieve Test IDs from cache file
        mlp_features_path = os.path.join(CACHE_DIR, "mlp_features.npz")
        if os.path.exists(mlp_features_path):
            data = np.load(mlp_features_path)
            test_ids = data["test_ids"]
        else:
            # Fallback if cache file missing (unlikely given flow)
            raise FileNotFoundError("MLP features cache not found for ID retrieval.")

        return test_ids, np.array(test_preds), best_val_auc
