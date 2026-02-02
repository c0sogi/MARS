import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.preprocessor import Preprocessor
from library.dataset import NotebookDataset, custom_collate_fn

# ==========================================
# Component Modules
# ==========================================


class ProjectionHead(nn.Module):
    """
    Projects 768-dim MPNet embeddings to a lower-dimensional latent space.
    """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ==========================================
# Main Model Architecture
# ==========================================


class CorrectedDCAN(nn.Module):
    """
    Corrected Dual-Context Anchor Network (DC-AN).

    Features:
    1. Symmetric Projection Heads for Code and Markdown.
    2. Code Branch: Transformer Encoder WITH Positional Encoding (Contextualized Skeleton).
    3. Markdown Branch: Transformer Encoder WITHOUT Positional Encoding (Set Transformer).
    4. Dynamic EOS Insertion: Appends a learnable EOS token to the end of valid code sequences.
    5. Interaction Head: Cross-Attention (Query=MD, Key=Code).
    """

    def __init__(self):
        super().__init__()
        self.config = Config

        # Projections
        self.code_proj = ProjectionHead(
            self.config.EMBEDDING_DIM, self.config.LATENT_DIM, self.config.DROPOUT
        )
        self.md_proj = ProjectionHead(
            self.config.EMBEDDING_DIM, self.config.LATENT_DIM, self.config.DROPOUT
        )

        # Learnable EOS Token
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.config.LATENT_DIM))

        # Code Branch (Sequential Context)
        self.pos_encoder = PositionalEncoding(
            self.config.LATENT_DIM,
            max_len=self.config.MAX_CODE_SEQ_LEN + 2,
            dropout=self.config.DROPOUT,
        )
        code_layer = nn.TransformerEncoderLayer(
            d_model=self.config.LATENT_DIM,
            nhead=self.config.NHEAD,
            dim_feedforward=self.config.LATENT_DIM * 4,
            dropout=self.config.DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.code_transformer = nn.TransformerEncoder(
            code_layer, num_layers=self.config.NUM_LAYERS
        )

        # Markdown Branch (Set Context - No Positional Encoding)
        md_layer = nn.TransformerEncoderLayer(
            d_model=self.config.LATENT_DIM,
            nhead=self.config.NHEAD,
            dim_feedforward=self.config.LATENT_DIM * 4,
            dropout=self.config.DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.md_transformer = nn.TransformerEncoder(
            md_layer, num_layers=self.config.NUM_LAYERS
        )

        # Output Scale
        self.scale = math.sqrt(self.config.LATENT_DIM)

    def forward(self, code_emb, md_emb, code_mask, md_mask, code_lens):
        """
        Args:
            code_emb: (B, L, 768)
            md_emb: (B, M, 768)
            code_mask: (B, L) - True for valid tokens
            md_mask: (B, M) - True for valid tokens
            code_lens: (B,) - Actual lengths of code sequences
        Returns:
            logits: (B, M, L+1) - Scores for placing MD before each code cell or at end.
        """
        B, L, _ = code_emb.shape
        _, M, _ = md_emb.shape
        device = code_emb.device

        # 1. Projection
        code_feat = self.code_proj(code_emb)  # (B, L, 512)
        md_feat = self.md_proj(md_emb)  # (B, M, 512)

        # 2. Dynamic EOS Insertion
        # We extend the sequence length by 1 to accommodate the EOS token.
        # We insert the EOS token at the index `code_lens[i]` for each batch item.

        # Initialize extended tensor with zeros
        extended_code = torch.zeros(B, L + 1, self.config.LATENT_DIM, device=device)
        # Copy original features
        extended_code[:, :L, :] = code_feat

        # Create scatter indices for EOS insertion
        # code_lens has shape (B,), we need (B, 1, Dim)
        scatter_indices = code_lens.view(B, 1, 1).expand(B, 1, self.config.LATENT_DIM)
        eos_expanded = self.eos_token.expand(B, 1, self.config.LATENT_DIM)

        # Insert EOS. Note: This overwrites whatever was at code_lens[i] (which was padding or 0)
        extended_code.scatter_(1, scatter_indices, eos_expanded)

        # Update Code Mask
        # The valid indices are now 0 to code_lens[i] (inclusive), total length code_lens[i] + 1
        # Shape (B, L+1)
        seq_indices = torch.arange(L + 1, device=device).unsqueeze(0).expand(B, L + 1)
        extended_code_mask = seq_indices <= code_lens.unsqueeze(1)

        # 3. Contextualization

        # Code Branch: Add Positional Encoding -> Transformer
        # Mask for Transformer (src_key_padding_mask takes True for PADDING)
        # extended_code_mask is True for VALID. So we invert.
        code_padding_mask = ~extended_code_mask

        code_ctx = self.pos_encoder(extended_code)
        code_ctx = self.code_transformer(
            code_ctx, src_key_padding_mask=code_padding_mask
        )

        # Markdown Branch: Set Transformer (No Positional Encoding)
        # Mask padding
        md_padding_mask = ~md_mask
        md_ctx = self.md_transformer(md_feat, src_key_padding_mask=md_padding_mask)

        # 4. Interaction Head (Cross-Attention)
        # Query: Markdown Context (B, M, D)
        # Key: Code Context (B, L+1, D)
        # Logits: (B, M, L+1)

        logits = torch.matmul(md_ctx, code_ctx.transpose(1, 2)) / self.scale

        # Mask out padding positions in Code (Keys)
        # extended_code_mask: (B, L+1). Unsqueeze to (B, 1, L+1)
        logits = logits.masked_fill(~extended_code_mask.unsqueeze(1), -1e9)

        return logits


# ==========================================
# Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        # Move batch to device
        code_emb = batch["code_emb"].to(device)
        md_emb = batch["md_emb"].to(device)
        code_mask = batch["code_mask"].to(device)
        md_mask = batch["md_mask"].to(device)
        code_lens = batch["code_lens"].to(device)
        labels = batch["labels"].to(device)  # (B, M)

        optimizer.zero_grad()

        # Forward
        logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)  # (B, M, L+1)

        # Flatten for loss
        # logits: (B*M, L+1)
        # labels: (B*M)
        B, M, L_plus_1 = logits.shape
        loss = criterion(logits.view(-1, L_plus_1), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / count if count > 0 else 0.0


def validate(model, loader, df_val_meta, device):
    model.eval()
    preds_list = []

    # We need to reconstruct the predicted order to compute Kendall Tau
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"]
            md_ids_batch = batch["md_ids"]

            code_emb = batch["code_emb"].to(device)
            md_emb = batch["md_emb"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["md_mask"].to(device)
            code_lens = batch["code_lens"].to(device)

            logits = model(
                code_emb, md_emb, code_mask, md_mask, code_lens
            )  # (B, M, L+1)
            probs = torch.softmax(logits, dim=-1)  # (B, M, L+1)

            # Compute Expected Index
            # indices: 0 to L
            L_plus_1 = probs.shape[-1]
            indices = torch.arange(L_plus_1, device=device).float()
            expected_pos = torch.sum(probs * indices, dim=-1)  # (B, M)

            expected_pos = expected_pos.cpu().numpy()

            for i, nb_id in enumerate(ids):
                # Get MD cell IDs for this notebook
                curr_md_ids = md_ids_batch[i]
                curr_scores = expected_pos[i, : len(curr_md_ids)]

                # Get Code cell IDs (we need to read them from metadata or assume order)
                # The validation metadata contains the ground truth order.
                # We need to separate code and md from the ground truth to reconstruct.
                # However, for metric computation, we just need the predicted order string.

                # We can't easily get code IDs from the batch tensors (they are embeddings).
                # Strategy: Load the notebook structure from metadata/json again or pass it through.
                # To be efficient, we will just store the (md_id, score) and merge later.

                for m_id, score in zip(curr_md_ids, curr_scores):
                    preds_list.append(
                        {
                            "id": nb_id,
                            "cell_id": m_id,
                            "rank_score": float(score)
                            - 0.5,  # Shift to align with code indices (0, 1, 2...)
                        }
                    )

    df_pred_scores = pd.DataFrame(preds_list)

    # Reconstruct Orders
    # We need to interleave MD cells into Code cells based on rank.
    # Code cell at index i has rank i.
    # MD cell has rank_score.

    # Load ground truth to get code cells
    df_val_meta_subset = df_val_meta[
        df_val_meta["id"].isin(df_pred_scores["id"].unique())
    ].copy()

    final_preds = []

    for _, row in df_val_meta_subset.iterrows():
        nb_id = row["id"]
        gt_order = row["cell_order"].split()

        # Identify code cells in order
        # We need to read the JSON to know which are code?
        # Or we can rely on the fact that we have the ground truth order.
        # But in inference we don't have GT order.
        # For validation, we can read the file or use the Preprocessor logic.
        # Let's read the file to identify code cells.
        try:
            # We can use the cached features to identify code cells if we loaded them differently,
            # but here we just read the json for simplicity in validation.
            # Actually, reading 20k files is slow.
            # Better: The Preprocessor cached data has cell_type.
            pass
        except:
            pass

    # Optimization: To avoid re-reading files, we can just use the fact that
    # the metric function `compute_kendall_tau` takes two ordered lists.
    # We need to produce the predicted list.
    # We know the set of cells.
    # Code cells have fixed relative order. MD cells are shuffled.
    # We assign:
    #   Code cell i: Rank i
    #   MD cell m: Rank predicted_score
    # Then sort.

    # We need the list of code cells for each notebook.
    # We can get this from the NotebookDataset if we modified it, but standard way:
    # Read the JSONs. Since Validation set is smaller (20k), it's feasible.
    # Or, we can assume the `cell_order` in train_orders.csv for Code is correct relative to each other.

    # Let's do this per notebook:
    # 1. Get all cells from GT.
    # 2. Identify Code/MD.
    # 3. Assign ranks.
    # 4. Sort.

    # To speed up, we'll process in memory using the df_pred_scores and reading JSONs on the fly is okay-ish.

    submission_rows = []

    # Pre-read all validation JSONs is too heavy.
    # We will iterate and read.

    nb_ids = df_pred_scores["id"].unique()

    # Create a dict for fast lookup
    pred_scores_map = df_pred_scores.set_index(["id", "cell_id"])[
        "rank_score"
    ].to_dict()

    for idx, row in df_val_meta_subset.iterrows():
        nb_id = row["id"]
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])

        with open(filepath, "r") as f:
            import json

            nb = json.load(f)

        cell_types = nb["cell_type"]
        # For validation, we have ground truth, but we must pretend we don't know the order of MD.
        # Code cells are in correct order in the JSON source usually?
        # In the competition data, "The code cells are in their original (correct) order."

        code_cells = [c for c in cell_types if cell_types[c] == "code"]
        md_cells = [c for c in cell_types if cell_types[c] == "markdown"]

        # In the provided JSONs, code cells might not be sorted by key, but the prompt says
        # "The code cells are in their original (correct) order." in the JSON *source*?
        # Actually, standard JSON is unordered. But Python dict preserves insertion order since 3.7.
        # The prompt says "train/... The code cells are in their original (correct) order."
        # So taking keys of code cells in order of appearance in JSON is correct.

        # Assign ranks
        cell_ranks = []
        for i, cid in enumerate(code_cells):
            cell_ranks.append((cid, float(i)))

        for cid in md_cells:
            score = pred_scores_map.get((nb_id, cid), 0.0)  # Default to 0 if missing
            cell_ranks.append((cid, score))

        # Sort
        cell_ranks.sort(key=lambda x: x[1])
        pred_order = " ".join([x[0] for x in cell_ranks])

        submission_rows.append({"id": nb_id, "cell_order": pred_order})

    df_pred = pd.DataFrame(submission_rows)

    score = compute_kendall_tau(df_val_meta_subset, df_pred)
    return score


def predict(model, loader, df_test_meta, device):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"]
            md_ids_batch = batch["md_ids"]

            code_emb = batch["code_emb"].to(device)
            md_emb = batch["md_emb"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["md_mask"].to(device)
            code_lens = batch["code_lens"].to(device)

            logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)
            probs = torch.softmax(logits, dim=-1)

            L_plus_1 = probs.shape[-1]
            indices = torch.arange(L_plus_1, device=device).float()
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

            for i, nb_id in enumerate(ids):
                curr_md_ids = md_ids_batch[i]
                curr_scores = expected_pos[i, : len(curr_md_ids)]

                for m_id, score in zip(curr_md_ids, curr_scores):
                    preds_list.append(
                        {"id": nb_id, "cell_id": m_id, "rank_score": float(score) - 0.5}
                    )

    df_pred_scores = pd.DataFrame(preds_list)

    # Generate Submission
    submission_rows = []
    pred_scores_map = df_pred_scores.set_index(["id", "cell_id"])[
        "rank_score"
    ].to_dict()

    for idx, row in df_test_meta.iterrows():
        nb_id = row["id"]
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])

        with open(filepath, "r") as f:
            import json

            nb = json.load(f)

        cell_types = nb["cell_type"]
        code_cells = [c for c in cell_types if cell_types[c] == "code"]
        md_cells = [c for c in cell_types if cell_types[c] == "markdown"]

        cell_ranks = []
        for i, cid in enumerate(code_cells):
            cell_ranks.append((cid, float(i)))

        for cid in md_cells:
            score = pred_scores_map.get((nb_id, cid), 0.0)
            cell_ranks.append((cid, score))

        cell_ranks.sort(key=lambda x: x[1])
        pred_order = " ".join([x[0] for x in cell_ranks])

        submission_rows.append({"id": nb_id, "cell_order": pred_order})

    return pd.DataFrame(submission_rows)


def run_training_and_inference():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Preprocessing
    print("Running Preprocessor...")
    preprocessor = Preprocessor()
    preprocessor.process_all(load_cached_data=True)

    # 2. Datasets
    print("Loading Datasets...")
    train_ds = NotebookDataset(Config.TRAIN_FEATURES_PATH, is_test=False)
    val_ds = NotebookDataset(Config.VAL_FEATURES_PATH, is_test=False)
    test_ds = NotebookDataset(Config.TEST_FEATURES_PATH, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model
    print("Initializing Model...")
    model = CorrectedDCAN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    best_score = -1.0
    patience_counter = 0
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, df_val_meta, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Kendall Tau: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved with score {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Inference
    print("Running Inference on Test Set...")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    df_submission = predict(model, test_loader, df_test_meta, device)

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")


# Execute the pipeline
if __name__ == "__main__":
    pass  # Guard to prevent auto-execution if imported, but we will call it below explicitly.

run_training_and_inference()
