import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
from library.config import Config
from library.utils import load_glove_embeddings, seed_everything
from library.data import get_dataloaders, get_test_dataloader, get_vocab

# =========================================================================
# Model Architecture
# =========================================================================


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attn_fc = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x: (Batch, Seq, Hidden)
        # scores: (Batch, Seq, 1)
        scores = self.attn_fc(x)
        weights = F.softmax(scores, dim=1)
        # context: (Batch, Hidden)
        context = torch.sum(x * weights, dim=1)
        return context


class CQCRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        embedding_matrix=None,
    ):
        super(CQCRNN, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix), requires_grad=False
            )

        # 2. Encoder
        # Input to GRU is Embed_Dim (Candidate) + Embed_Dim (Question Context)
        self.gru = nn.GRU(
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 3. Pooling
        self.attn_pooling = AttentionPooling(hidden_dim * 2)

        # 4. Heads
        # Long Answer Ranking (Binary)
        self.long_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Short Answer Spans (Start/End)
        # Maps hidden state at each step to a logit
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

        # Yes/No Classification
        self.yn_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, Config.NUM_CLASSES_YES_NO),
        )

    def forward(self, q_input, c_input):
        # q_input: (B, Q_Len)
        # c_input: (B, C_Len)

        # Embeddings
        q_emb = self.embedding(q_input)  # (B, Q, E)
        c_emb = self.embedding(c_input)  # (B, C, E)

        # Question Compression (Mean Pooling)
        # Mask padding (assuming 0 is pad)
        q_mask = (q_input != 0).unsqueeze(-1).float()  # (B, Q, 1)
        q_sum = torch.sum(q_emb * q_mask, dim=1)
        q_len = torch.sum(q_mask, dim=1).clamp(min=1e-9)
        q_ctx = q_sum / q_len  # (B, E)

        # Context Fusion
        # Expand q_ctx to match candidate length
        seq_len = c_input.size(1)
        q_ctx_expanded = q_ctx.unsqueeze(1).expand(-1, seq_len, -1)  # (B, C, E)

        # Concatenate: [Candidate Token; Global Question Context]
        rnn_input = torch.cat([c_emb, q_ctx_expanded], dim=2)  # (B, C, 2E)

        # Encoder
        rnn_out, _ = self.gru(rnn_input)  # (B, C, 2H)

        # Document Representation
        doc_vec = self.attn_pooling(rnn_out)  # (B, 2H)

        # Heads
        long_logits = self.long_head(doc_vec)  # (B, 1)

        start_logits = self.start_head(rnn_out).squeeze(-1)  # (B, C)
        end_logits = self.end_head(rnn_out).squeeze(-1)  # (B, C)

        yn_logits = self.yn_head(doc_vec)  # (B, Num_Classes)

        return {
            "long_logits": long_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yn_logits": yn_logits,
        }


# =========================================================================
# Training Logic
# =========================================================================


def calculate_loss(outputs, batch):
    # Long Answer: BCEWithLogits
    loss_long = F.binary_cross_entropy_with_logits(
        outputs["long_logits"].squeeze(-1), batch["label_long"]
    )

    # Short Answer: CrossEntropy (Sparse)
    # Only calculate span loss for positive long answers to avoid noise from negatives
    # However, standard approach often trains on all or masks negatives.
    # Here we train on all but negatives point to index 0 (NULL).
    loss_start = F.cross_entropy(outputs["start_logits"], batch["label_start"])
    loss_end = F.cross_entropy(outputs["end_logits"], batch["label_end"])

    # Yes/No: CrossEntropy
    loss_yn = F.cross_entropy(outputs["yn_logits"], batch["label_yn"])

    total_loss = (
        Config.WEIGHT_LONG_ANSWER * loss_long
        + Config.WEIGHT_SHORT_SPAN * (loss_start + loss_end)
        + Config.WEIGHT_YES_NO * loss_yn
    )

    return (
        total_loss,
        loss_long.item(),
        loss_start.item(),
        loss_end.item(),
        loss_yn.item(),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()

        outputs = model(batch["q_input"], batch["c_input"])
        loss, _, _, _, _ = calculate_loss(outputs, batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct_long = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch["q_input"], batch["c_input"])
            loss, _, _, _, _ = calculate_loss(outputs, batch)
            running_loss += loss.item()

            # Simple accuracy for long answer
            preds = torch.sigmoid(outputs["long_logits"]).squeeze(-1) > 0.5
            targets = batch["label_long"] > 0.5
            correct_long += (preds == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = running_loss / len(loader)
    acc_long = correct_long / total_samples if total_samples > 0 else 0
    return avg_loss, acc_long


def run_training():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data
    vocab = get_vocab(load_cached_data=True)
    train_loader, val_loader = get_dataloaders(vocab, load_cached_data=True)
    embeddings = load_glove_embeddings(vocab.stoi, load_cached_data=True)

    # 2. Model
    model = CQCRNN(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        embedding_matrix=embeddings,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 3. Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Long Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


# =========================================================================
# Inference Logic
# =========================================================================


def get_candidate_offsets_map(jsonl_path):
    """
    Parses the raw JSONL file to map example_id -> list of candidate offsets.
    This is necessary because the processed features dropped this info.
    """
    offset_map = {}
    if not os.path.exists(jsonl_path):
        return offset_map

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ex_id = str(entry["example_id"])
            candidates = entry.get("long_answer_candidates", [])
            # Store list of dicts: {'s': start, 'e': end}
            offset_map[ex_id] = [
                {"s": c["start_token"], "e": c["end_token"]} for c in candidates
            ]
    return offset_map


def generate_submission():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data & Model
    vocab = get_vocab(load_cached_data=True)
    test_loader = get_test_dataloader(vocab, load_cached_data=True)

    model = CQCRNN(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No trained model found. Generating random predictions.")

    model.eval()

    # 2. Load Offsets for reconstruction
    print("Loading candidate offsets from raw test file...")
    offset_map = get_candidate_offsets_map(Config.TEST_DATA_PATH)

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs
            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)

            # Forward
            outputs = model(q_input, c_input)

            # Move to CPU
            long_scores = torch.sigmoid(outputs["long_logits"]).cpu().numpy().flatten()
            start_logits = outputs["start_logits"].cpu().numpy()
            end_logits = outputs["end_logits"].cpu().numpy()
            yn_logits = outputs["yn_logits"].cpu().numpy()

            # Reconstruct by example
            example_ids = batch["example_ids"]
            counts = batch["candidate_counts"]

            current_idx = 0
            for i, ex_id in enumerate(example_ids):
                count = counts[i]
                if count == 0:
                    # Edge case: no candidates
                    results.append(f"{ex_id}_long,")
                    results.append(f"{ex_id}_short,")
                    continue

                # Slice outputs for this example
                sl_scores = long_scores[current_idx : current_idx + count]
                sl_start = start_logits[current_idx : current_idx + count]
                sl_end = end_logits[current_idx : current_idx + count]
                sl_yn = yn_logits[current_idx : current_idx + count]

                # 1. Select Best Candidate
                best_cand_idx = np.argmax(sl_scores)
                best_score = sl_scores[best_cand_idx]

                # Default predictions
                pred_long = ""
                pred_short = ""

                # Threshold check
                if best_score >= Config.LONG_ANSWER_THRESHOLD:
                    # Get offsets
                    if ex_id in offset_map and best_cand_idx < len(offset_map[ex_id]):
                        cand_offsets = offset_map[ex_id][best_cand_idx]
                        c_start_doc = cand_offsets["s"]
                        c_end_doc = cand_offsets["e"]

                        # Set Long Answer
                        pred_long = f"{c_start_doc}:{c_end_doc}"

                        # 2. Determine Short Answer
                        # Get relative indices
                        s_idx = np.argmax(sl_start[best_cand_idx])
                        e_idx = np.argmax(sl_end[best_cand_idx])
                        yn_idx = np.argmax(sl_yn[best_cand_idx])

                        # Check Yes/No first (Class 1=YES, 2=NO, 0=NONE)
                        if yn_idx == 1:
                            pred_short = "YES"
                        elif yn_idx == 2:
                            pred_short = "NO"
                        else:
                            # Span prediction
                            # Index 0 is NULL/CLS token. Valid span must be > 0 and start <= end
                            if s_idx > 0 and e_idx > 0 and s_idx <= e_idx:
                                # Map relative to document absolute
                                # s_idx=1 corresponds to first token of candidate
                                abs_s = c_start_doc + s_idx - 1
                                abs_e = c_start_doc + e_idx - 1

                                # Sanity check bounds
                                if abs_e < c_end_doc:
                                    pred_short = f"{abs_s}:{abs_e}"

                results.append(f"{ex_id}_long,{pred_long}")
                results.append(f"{ex_id}_short,{pred_short}")

                current_idx += count

    # Write submission
    with open(Config.SUBMISSION_PATH, "w") as f:
        f.write("example_id,PredictionString\n")
        f.write("\n".join(results))

    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # If run directly, perform training then inference
    run_training()
    generate_submission()
