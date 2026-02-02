import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
from transformers import AutoModel
from library.config import Config
from library.utils import compute_spearmanr


class GranularSiameseModel(nn.Module):
    """
    Granular Siamese DeBERTa model with Segment-Aware Pooling and Metadata Fusion.

    Architecture:
    1. Shared DeBERTa Backbone for Question and Answer streams.
    2. Question Stream: Segment-Aware Pooling to extract distinct Title and Body vectors.
    3. Answer Stream: Standard Mean Pooling.
    4. Granular Interactions: Explicit interaction features (diff, prod) for Title-Answer and Body-Answer.
    5. Metadata Fusion: Learnable embeddings for categorical features.
    6. Unified MLP Head: Predicts all 30 targets.
    """

    def __init__(self, cat_dims, embedding_dim=16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.hidden_size = Config.HIDDEN_SIZE

        # Metadata Embeddings
        # cat_dims is a list of cardinalities for [category, host]
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=dim, embedding_dim=embedding_dim)
                for dim in cat_dims
            ]
        )

        # Calculate input dimension for the MLP head
        # Interaction Features:
        # Title-Answer: [u_title, v_answer, abs_diff, prod] -> 4 * Hidden
        # Body-Answer:  [u_body, v_answer, abs_diff, prod]  -> 4 * Hidden
        # Total Text Features: 8 * Hidden
        text_feat_dim = 8 * self.hidden_size

        # Metadata Features: Sum of embedding dimensions
        meta_feat_dim = len(cat_dims) * embedding_dim

        input_dim = text_feat_dim + meta_feat_dim

        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, Config.NUM_TARGETS),
        )

    def masked_mean(self, hidden, mask):
        """
        Computes the mean of hidden states, ignoring masked tokens.

        Args:
            hidden: [Batch, Seq, Hidden]
            mask: [Batch, Seq] (1 for keep, 0 for ignore)
        """
        # Expand mask to [Batch, Seq, Hidden]
        mask_expanded = mask.unsqueeze(-1).expand(hidden.size())

        # Sum embeddings over sequence dimension
        sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)

        # Count valid tokens (clamp to avoid division by zero)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        q_token_type_ids,
        q_title_mask,
        q_body_mask,
        a_input_ids,
        a_attention_mask,
        a_token_type_ids,
        cat_feats,
    ):

        # --- Question Stream ---
        q_out = self.backbone(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
        )
        q_last = q_out.last_hidden_state  # [Batch, Seq, Hidden]

        # Segment-Aware Pooling
        # Extract specific representations for Title and Body using pre-computed masks
        u_title = self.masked_mean(q_last, q_title_mask)
        u_body = self.masked_mean(q_last, q_body_mask)

        # --- Answer Stream ---
        a_out = self.backbone(
            input_ids=a_input_ids,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
        )
        v_answer = self.masked_mean(a_out.last_hidden_state, a_attention_mask)

        # --- Interactions ---
        # 1. Title-Answer Interaction
        i_title = torch.cat(
            [u_title, v_answer, torch.abs(u_title - v_answer), u_title * v_answer],
            dim=1,
        )

        # 2. Body-Answer Interaction
        i_body = torch.cat(
            [u_body, v_answer, torch.abs(u_body - v_answer), u_body * v_answer], dim=1
        )

        # --- Metadata Fusion ---
        cat_embeds = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            # cat_feats is [Batch, Num_Cats]
            cat_embeds.append(emb_layer(cat_feats[:, i]))
        meta_vec = torch.cat(cat_embeds, dim=1)

        # --- Final Concatenation ---
        features = torch.cat([i_title, i_body, meta_vec], dim=1)

        # --- Prediction ---
        logits = self.head(features)

        # Return probabilities in range [0, 1]
        return torch.sigmoid(logits)


def train_fn(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0

    for batch in loader:
        # Unpack batch (11 elements)
        (
            qa_ids,
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            q_title_mask,
            q_body_mask,
            a_input_ids,
            a_attention_mask,
            a_token_type_ids,
            cat_feats,
            targets,
        ) = [b.to(device) for b in batch]

        optimizer.zero_grad()

        preds = model(
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            q_title_mask,
            q_body_mask,
            a_input_ids,
            a_attention_mask,
            a_token_type_ids,
            cat_feats,
        )

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_fn(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and mean column-wise Spearman correlation.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            (
                qa_ids,
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                q_title_mask,
                q_body_mask,
                a_input_ids,
                a_attention_mask,
                a_token_type_ids,
                cat_feats,
                targets,
            ) = [b.to(device) for b in batch]

            preds = model(
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                q_title_mask,
                q_body_mask,
                a_input_ids,
                a_attention_mask,
                a_token_type_ids,
                cat_feats,
            )

            loss = criterion(preds, targets)
            total_loss += loss.item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    spearman = compute_spearmanr(all_preds, all_targets)
    return total_loss / len(loader), spearman


def predict_fn(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_qa_ids = []

    with torch.no_grad():
        for batch in loader:
            # Note: Test loader has dummy targets, we ignore them
            (
                qa_ids,
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                q_title_mask,
                q_body_mask,
                a_input_ids,
                a_attention_mask,
                a_token_type_ids,
                cat_feats,
                _,
            ) = [b.to(device) for b in batch]

            preds = model(
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                q_title_mask,
                q_body_mask,
                a_input_ids,
                a_attention_mask,
                a_token_type_ids,
                cat_feats,
            )

            all_preds.append(preds.cpu().numpy())
            all_qa_ids.extend(qa_ids.cpu().numpy())

    return np.vstack(all_preds), np.array(all_qa_ids)


def run_experiment(train_loader, val_loader, test_loader, cat_dims):
    """
    Main execution function:
    1. Initializes model, optimizer, and criterion.
    2. Runs training loop with early stopping.
    3. Saves best model.
    4. Generates and saves submission file.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = GranularSiameseModel(cat_dims).to(device)

    # Differential Learning Rates
    # Lower LR for backbone to preserve pre-trained knowledge
    # Higher LR for new head and embeddings to converge faster
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
        {"params": model.cat_embeddings.parameters(), "lr": Config.LR_HEAD},
        {"params": model.head.parameters(), "lr": Config.LR_HEAD},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY
    )

    # Use BCELoss as targets are continuous probabilities in [0, 1]
    criterion = nn.BCELoss()

    best_score = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = eval_fn(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Spearman: {val_score:.6f}"
        )

        # Save best model based on Spearman correlation
        if val_score > best_score:
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_wts, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Val Spearman: {best_score:.6f}")

    # --- Inference ---
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    print("Generating predictions on test set...")
    test_preds, test_ids = predict_fn(model, test_loader, device)

    # Create Submission DataFrame
    sub_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    sub_df.insert(0, "qa_id", test_ids)

    # Save submission
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
