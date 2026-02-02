import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from library.config import Config


class DecomposableAttentionRanker(nn.Module):
    """
    Decomposable Attention Model for Natural Language Inference/Ranking.
    Based on Parikh et al. (2016).
    """

    def __init__(self, embedding_matrix=None):
        super(DecomposableAttentionRanker, self).__init__()

        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.hidden_size = Config.RANKER_HIDDEN_SIZE
        self.dropout_prob = Config.RANKER_DROPOUT

        # 1. Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        if embedding_matrix is not None:
            # Initialize with pre-trained embeddings
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )

        # 2. Attend Module (Projection)
        # Projects input embeddings to hidden size for attention score computation
        self.attend_project = nn.Linear(self.embed_dim, self.hidden_size, bias=False)

        # 3. Compare Module (Feed Forward Network)
        # Processes concatenated [original, aligned] representations
        self.compare_ffn = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )

        # 4. Aggregate Module (Feed Forward Network)
        # Processes summed representations from both sequences
        self.aggregate_ffn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, q_indices, doc_indices):
        """
        Args:
            q_indices: (Batch, Q_Len)
            doc_indices: (Batch, D_Len)
        Returns:
            logits: (Batch, 1)
        """
        # Create masks (1 for valid token, 0 for pad)
        q_mask = (q_indices != 0).float()  # (B, Q)
        d_mask = (doc_indices != 0).float()  # (B, D)

        # Embed inputs
        q_embed = self.embedding(q_indices)  # (B, Q, E)
        d_embed = self.embedding(doc_indices)  # (B, D, E)

        # --- Attend Phase ---
        # Project embeddings
        q_proj = self.attend_project(q_embed)  # (B, Q, H)
        d_proj = self.attend_project(d_embed)  # (B, D, H)

        # Compute Attention Scores: E = Q' * D'^T
        scores = torch.bmm(q_proj, d_proj.transpose(1, 2))  # (B, Q, D)

        # Apply masks to scores before softmax
        # For softmax over D (dim 2), mask invalid D tokens
        d_mask_expanded = d_mask.unsqueeze(1)  # (B, 1, D)
        scores_q = scores.masked_fill(d_mask_expanded == 0, -1e9)

        # For softmax over Q (dim 1), mask invalid Q tokens
        q_mask_expanded = q_mask.unsqueeze(2)  # (B, Q, 1)
        scores_d = scores.masked_fill(q_mask_expanded == 0, -1e9)

        # Compute Attention Weights
        alpha = F.softmax(scores_q, dim=2)  # (B, Q, D) - Align D to Q
        beta = F.softmax(scores_d, dim=1)  # (B, Q, D) - Align Q to D

        # Compute Aligned Representations
        # For each Q token, weighted sum of D tokens
        d_aligned = torch.bmm(alpha, d_embed)  # (B, Q, D) x (B, D, E) -> (B, Q, E)

        # For each D token, weighted sum of Q tokens
        # Transpose beta to (B, D, Q)
        q_aligned = torch.bmm(
            beta.transpose(1, 2), q_embed
        )  # (B, D, Q) x (B, Q, E) -> (B, D, E)

        # --- Compare Phase ---
        # Concatenate original and aligned representations
        q_compare_input = torch.cat([q_embed, d_aligned], dim=2)  # (B, Q, 2E)
        d_compare_input = torch.cat([d_embed, q_aligned], dim=2)  # (B, D, 2E)

        # Pass through Compare FFN
        q_compare = self.compare_ffn(q_compare_input)  # (B, Q, H)
        d_compare = self.compare_ffn(d_compare_input)  # (B, D, H)

        # --- Aggregate Phase ---
        # Apply masks to zero out padding contributions before summing
        q_compare = q_compare * q_mask.unsqueeze(2)
        d_compare = d_compare * d_mask.unsqueeze(2)

        # Sum over sequence length
        v1 = q_compare.sum(dim=1)  # (B, H)
        v2 = d_compare.sum(dim=1)  # (B, H)

        # Concatenate aggregated vectors
        v_agg = torch.cat([v1, v2], dim=1)  # (B, 2H)

        # Final prediction
        logits = self.aggregate_ffn(v_agg)  # (B, 1)

        return logits


def train_ranker(train_loader, val_loader, embedding_matrix, device):
    """
    Trains the Ranker model with Early Stopping.
    """
    model = DecomposableAttentionRanker(embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting Ranker training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Training Loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            q_indices = batch["q_indices"].to(device)
            doc_indices = batch["doc_indices"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)  # Ensure shape (B, 1)

            optimizer.zero_grad()
            logits = model(q_indices, doc_indices)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # Metrics
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += batch_size

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                q_indices = batch["q_indices"].to(device)
                doc_indices = batch["doc_indices"].to(device)
                labels = batch["labels"].to(device).unsqueeze(1)

                logits = model(q_indices, doc_indices)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {avg_train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {avg_val_loss} | Val Acc: {val_acc}"
        )

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
            print(f"  New best model saved to {Config.RANKER_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return model
