import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional, Any

from library.config import Config
from library.utils import seed_everything, print_metric, ensure_directory
from library.dataset import get_dataloaders


class DualQueryAttention(nn.Module):
    """
    Computes attention over the history sequence using a specific query (Title or Body).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.scale = input_dim**-0.5
        # We assume Query, Key, Value all have the same input dimension (SBERT dim)
        # We don't project Q, K, V here to keep it "Raw" as per instructions,
        # or we can use a simple linear transform. The prompt says "Raw SBERT embeddings",
        # but usually attention involves projections. However, "Branch 1... Processes Raw SBERT... No early projection"
        # implies we might operate directly. Standard attention usually projects.
        # Given "Dual-Query History Attention... Input: Sequence of Raw SBERT embeddings",
        # we will implement standard Scaled Dot-Product Attention without learnable projections
        # inside the attention mechanism itself, or minimal projections if needed for dimension matching.
        # Since Q and K are both 384-dim SBERT, we can do direct dot product.

        # However, to allow the model to learn *what* to attend to, learnable projections are standard.
        # We will add linear projections for Q, K, V to a common attention dimension.
        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)

    def forward(
        self, query: torch.Tensor, history: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, D)
            history: (B, L, D)
            mask: (B, L) - 1 for valid, 0 for padding
        Returns:
            context: (B, D)
        """
        # Project
        Q = self.W_q(query).unsqueeze(1)  # (B, 1, D)
        K = self.W_k(history)  # (B, L, D)
        V = self.W_v(history)  # (B, L, D)

        # Dot Product
        # (B, 1, D) @ (B, D, L) -> (B, 1, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Masking
        # Mask is (B, L). We need to broadcast to (B, 1, L).
        # Where mask is 0, set scores to -inf
        mask_expanded = mask.unsqueeze(1)  # (B, 1, L)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, L)

        # Weighted Sum
        # (B, 1, L) @ (B, L, D) -> (B, 1, D)
        context = torch.matmul(attn_weights, V)

        return context.squeeze(1)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Modulates input x based on conditioning z.
    """

    def __init__(self, x_dim: int, z_dim: int):
        super().__init__()
        self.scale_gen = nn.Linear(z_dim, x_dim)
        self.shift_gen = nn.Linear(z_dim, x_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vector to modulate (B, x_dim)
            z: Conditioning vector (B, z_dim)
        """
        gamma = self.scale_gen(z)
        beta = self.shift_gen(z)

        # Output = (1 + gamma) * x + beta
        return (1.0 + gamma) * x + beta


class FiLMDualAttentionMLP(nn.Module):
    def __init__(self, metadata_dim: int):
        super().__init__()

        emb_dim = Config.EMBEDDING_DIM
        hidden_dim = Config.HIDDEN_DIM
        dropout_rate = Config.DROPOUT_RATE
        dense_dropout = Config.DENSE_DROPOUT

        # --- Branch 5: Metadata (Conditioning Source) ---
        # Input: Metadata + TopK + Consistency
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim), nn.ReLU(), nn.Dropout(dense_dropout)
        )

        # --- Semantic Branches ---
        # Branch 1: Title
        self.title_proj = nn.Linear(emb_dim, hidden_dim)

        # Branch 2: Body
        self.body_proj = nn.Linear(emb_dim, hidden_dim)

        # Branch 3: Dual-Query History Attention
        self.attn_title_hist = DualQueryAttention(emb_dim)
        self.attn_body_hist = DualQueryAttention(emb_dim)
        self.hist_proj = nn.Linear(emb_dim, hidden_dim)  # Project attended context

        # Branch 4: Global Persona Injection
        self.persona_proj = nn.Linear(emb_dim, hidden_dim)

        # --- Fusion ---
        # We concatenate:
        # 1. Title Proj (H)
        # 2. Body Proj (H)
        # 3. Attended History (Title Query) (H)
        # 4. Attended History (Body Query) (H)
        # 5. Persona Centroid (H)
        self.semantic_dim = hidden_dim * 5

        # FiLM Layer
        # Modulates semantic_dim based on metadata hidden_dim
        self.film = FiLMLayer(x_dim=self.semantic_dim, z_dim=hidden_dim)

        # Final Classifier
        # Input: Modulated Semantics + Metadata (Skip Connection)
        final_input_dim = self.semantic_dim + hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        title_emb: torch.Tensor,
        body_emb: torch.Tensor,
        history_seq: torch.Tensor,
        history_mask: torch.Tensor,
        history_cent: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:

        # 1. Encode Metadata (z)
        z = self.metadata_encoder(metadata)  # (B, H)

        # 2. Process Semantics
        # Title & Body
        h_title = self.title_proj(title_emb)  # (B, H)
        h_body = self.body_proj(body_emb)  # (B, H)

        # History Attention
        # Context 1: Query = Title
        ctx_title = self.attn_title_hist(title_emb, history_seq, history_mask)
        h_ctx_title = self.hist_proj(ctx_title)  # (B, H)

        # Context 2: Query = Body
        ctx_body = self.attn_body_hist(body_emb, history_seq, history_mask)
        h_ctx_body = self.hist_proj(ctx_body)  # (B, H)

        # Persona
        h_persona = self.persona_proj(history_cent)  # (B, H)

        # Concatenate Semantics
        x_sem = torch.cat(
            [h_title, h_body, h_ctx_title, h_ctx_body, h_persona], dim=1
        )  # (B, 5H)

        # 3. FiLM Modulation
        x_modulated = self.film(x_sem, z)

        # 4. Final Fusion (Skip connection from z)
        fusion = torch.cat([x_modulated, z], dim=1)

        # 5. Classification
        logits = self.classifier(fusion)
        return logits.squeeze(1)


class MLPPipeline:
    def __init__(self, metadata_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FiLMDualAttentionMLP(metadata_dim).to(self.device)
        self.model_path = os.path.join(Config.CACHE_DIR, "best_mlp.pth")

        print(f"MLP initialized on {self.device}. Metadata Dim: {metadata_dim}")

    def train(self, train_loader, val_loader):
        seed_everything()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

        for epoch in range(Config.NUM_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # Move to device
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist_seq = batch["history_seq"].to(self.device)
                hist_mask = batch["history_mask"].to(self.device)
                hist_cent = batch["history_cent"].to(self.device)
                meta = batch["metadata"].to(self.device)
                target = batch["target"].to(self.device)

                optimizer.zero_grad()

                logits = self.model(title, body, hist_seq, hist_mask, hist_cent, meta)
                loss = criterion(logits, target)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * title.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            val_auc, val_loss = self.evaluate(val_loader, criterion)

            # print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.6f}")

            # Early Stopping & Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_auc}"
                    )
                    break

        print_metric("MLP_Best_Validation_AUC", best_auc)

        # Load best model
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )

    def evaluate(self, loader, criterion=None):
        self.model.eval()
        targets = []
        preds = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist_seq = batch["history_seq"].to(self.device)
                hist_mask = batch["history_mask"].to(self.device)
                hist_cent = batch["history_cent"].to(self.device)
                meta = batch["metadata"].to(self.device)

                logits = self.model(title, body, hist_seq, hist_mask, hist_cent, meta)
                probs = torch.sigmoid(logits)

                preds.extend(probs.cpu().numpy())

                if "target" in batch:
                    target = batch["target"].to(self.device)
                    targets.extend(target.cpu().numpy())
                    if criterion:
                        loss = criterion(logits, target)
                        total_loss += loss.item() * title.size(0)

        if criterion and len(loader.dataset) > 0:
            total_loss /= len(loader.dataset)

        auc = 0.0
        if len(targets) > 0:
            try:
                auc = roc_auc_score(targets, preds)
            except ValueError:
                auc = 0.5  # Handle case with single class in batch if strictly needed, though unlikely in val

        return auc, total_loss

    def predict(self, loader):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist_seq = batch["history_seq"].to(self.device)
                hist_mask = batch["history_mask"].to(self.device)
                hist_cent = batch["history_cent"].to(self.device)
                meta = batch["metadata"].to(self.device)

                logits = self.model(title, body, hist_seq, hist_mask, hist_cent, meta)
                probs = torch.sigmoid(logits)
                preds.extend(probs.cpu().numpy())

        return np.array(preds)


def run_mlp_pipeline(
    mlp_data: Dict[str, Any], force_retrain: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the MLP pipeline.
    """
    # 1. Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        mlp_data,
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Use a small number of workers
    )

    # 2. Determine Metadata Dimension from a sample
    # We can peek at the dataset
    sample_batch = next(iter(train_loader))
    metadata_dim = sample_batch["metadata"].shape[1]

    # 3. Initialize Pipeline
    pipeline = MLPPipeline(metadata_dim)

    # 4. Train or Load
    model_exists = os.path.exists(pipeline.model_path)

    if force_retrain or not model_exists:
        pipeline.train(train_loader, val_loader)
    else:
        print("Loading MLP model from cache...")
        pipeline.model.load_state_dict(
            torch.load(pipeline.model_path, map_location=pipeline.device)
        )

    # 5. Predict
    print("Generating predictions with MLP...")
    val_probs = pipeline.predict(val_loader)
    test_probs = pipeline.predict(test_loader)

    return val_probs, test_probs
