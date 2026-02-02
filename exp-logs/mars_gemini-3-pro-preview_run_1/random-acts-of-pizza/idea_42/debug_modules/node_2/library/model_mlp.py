import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from library import config


class DualQueryAttention(nn.Module):
    """
    Attends to user history using two distinct queries: Request Title and Request Body.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Batch_first=True ensures inputs are (Batch, Seq, Dim)
        self.mha_title = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.mha_body = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

    def forward(self, title, body, history, key_padding_mask):
        """
        Args:
            title: (B, D)
            body: (B, D)
            history: (B, S, D)
            key_padding_mask: (B, S) - True where padding exists
        Returns:
            attn_title: (B, D)
            attn_body: (B, D)
        """
        # Prepare queries: (B, 1, D)
        q_title = title.unsqueeze(1)
        q_body = body.unsqueeze(1)

        # Multi-Head Attention
        # Output is (B, 1, D)
        out_title, _ = self.mha_title(
            q_title, history, history, key_padding_mask=key_padding_mask
        )
        out_body, _ = self.mha_body(
            q_body, history, history, key_padding_mask=key_padding_mask
        )

        return out_title.squeeze(1), out_body.squeeze(1)


class PersonaAwareSkipGatedMLP(nn.Module):
    """
    Hybrid Neural Network that fuses Semantic Persona, Interaction History, and Numerical Metadata
    via a Skip-Gated mechanism.
    """

    def __init__(
        self,
        meta_dim,
        embed_dim=config.EMBEDDING_DIM,
        hidden_dims=config.MLP_HIDDEN_DIMS,
        dropout_emb=config.MLP_DROPOUT_EMB,
        dropout_dense=config.MLP_DROPOUT_DENSE,
    ):
        super().__init__()

        self.dropout_emb = nn.Dropout(dropout_emb)
        self.dropout_dense = nn.Dropout(dropout_dense)

        # 1. Dual Query Attention Branch
        self.dual_attention = DualQueryAttention(embed_dim, dropout=dropout_dense)

        # Dimensions Calculation
        # Semantic Vector = [Title, Body, Attn_Title, Attn_Body, Align_Title, Align_Body]
        # Align scalars are 1D each.
        self.semantic_dim = (embed_dim * 4) + 2

        # Gate Source = [Metadata, Persona]
        self.gate_source_dim = meta_dim + embed_dim

        # 2. Gate Network
        # Learns how much to trust the semantic signals based on metadata/persona
        self.gate_net = nn.Sequential(
            nn.Linear(self.gate_source_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.semantic_dim),
            nn.Sigmoid(),
        )

        # 3. Fusion Layer
        # Concatenates [Gated_Semantic, Metadata, Persona]
        # Skip connection ensures Metadata/Persona are always available to the head
        fusion_input_dim = self.semantic_dim + meta_dim + embed_dim

        # 4. MLP Head
        layers = []
        in_dim = fusion_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_dense))
            in_dim = h_dim

        # Final Logit
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, title, body, history, history_mask, persona, metadata):
        # Apply Dropout to Embeddings
        title = self.dropout_emb(title)
        body = self.dropout_emb(body)
        history = self.dropout_emb(history)
        persona = self.dropout_emb(persona)

        # 1. Compute Attended Contexts
        attn_title, attn_body = self.dual_attention(title, body, history, history_mask)

        # 2. Compute Alignment Scalars (Cosine Similarity)
        # Normalize vectors for cosine calculation
        t_norm = F.normalize(title, p=2, dim=1)
        b_norm = F.normalize(body, p=2, dim=1)
        p_norm = F.normalize(persona, p=2, dim=1)

        # Dot product (B, D) * (B, D) -> (B, 1)
        align_title = (t_norm * p_norm).sum(dim=1, keepdim=True)
        align_body = (b_norm * p_norm).sum(dim=1, keepdim=True)

        # 3. Construct Semantic Vector
        semantic_vector = torch.cat(
            [title, body, attn_title, attn_body, align_title, align_body], dim=1
        )

        # 4. Construct Gate Source
        gate_source = torch.cat([metadata, persona], dim=1)

        # 5. Compute Gate
        gate = self.gate_net(gate_source)

        # 6. Apply Gate (Element-wise)
        gated_semantic = semantic_vector * gate

        # 7. Final Fusion (Skip Connection)
        fusion_vec = torch.cat([gated_semantic, metadata, persona], dim=1)

        # 8. Prediction
        logits = self.mlp_head(fusion_vec)
        return logits


def train_mlp(train_loader, val_loader, meta_dim, device):
    """
    Trains the PersonaAwareSkipGatedMLP with Early Stopping.
    """
    print(f"Initializing MLP with Meta Dim: {meta_dim} on {device}")

    model = PersonaAwareSkipGatedMLP(meta_dim=meta_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.MLP_LEARNING_RATE,
        weight_decay=config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.MLP_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []

        for batch, targets in train_loader:
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(
                batch["title_emb"],
                batch["body_emb"],
                batch["history_seq"],
                batch["history_mask"],
                batch["persona_centroid"],
                batch["dense_metadata"],
            )

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(all_targets, all_preds)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch, targets in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                targets = targets.to(device).unsqueeze(1)

                logits = model(
                    batch["title_emb"],
                    batch["body_emb"],
                    batch["history_seq"],
                    batch["history_mask"],
                    batch["persona_centroid"],
                    batch["dense_metadata"],
                )

                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{config.MLP_EPOCHS} - Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.MLP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_val_auc}"
                )
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_mlp(model, loader, device):
    """
    Generates probability predictions for a given loader.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle dataset returning (batch, label) or (batch)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                batch["title_emb"],
                batch["body_emb"],
                batch["history_seq"],
                batch["history_mask"],
                batch["persona_centroid"],
                batch["dense_metadata"],
            )

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    return np.array(preds)
