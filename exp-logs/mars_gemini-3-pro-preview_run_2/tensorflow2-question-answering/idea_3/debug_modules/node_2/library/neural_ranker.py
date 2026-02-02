import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from library.config import Config
from library.text_utils import TextUtils
from library.data_factory import DataFactory

# -----------------------------------------------------------------------------
# Neural Network Architecture
# -----------------------------------------------------------------------------


class GatedConvEncoder(nn.Module):
    """
    1D Convolutional Encoder with Gated Linear Units (GLU) and Max-Pooling.
    """

    def __init__(
        self,
        embedding_dim: int,
        kernel_sizes: List[int],
        num_filters: int,
        dropout: float,
    ):
        super(GatedConvEncoder, self).__init__()
        # Convolutions
        # We output 2 * num_filters channels to support GLU (split into content and gate)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=2 * num_filters,
                    kernel_size=k,
                    padding=k // 2,  # Padding to maintain sequence length roughly
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [batch, seq_len, emb_dim]
        # Permute for Conv1d: [batch, emb_dim, seq_len]
        x = x.transpose(1, 2)

        pooled_outputs = []
        for conv in self.convs:
            # Convolution
            out = conv(x)  # [batch, 2*filters, L_out]

            # GLU Mechanism: Split channels into Content (A) and Gate (B)
            A, B = out.chunk(2, dim=1)
            gated = A * torch.sigmoid(B)  # [batch, filters, L_out]

            # Global Max Pooling over the sequence dimension (dim=2)
            # Result: [batch, filters]
            pooled = F.max_pool1d(gated, kernel_size=gated.size(2)).squeeze(2)
            pooled_outputs.append(pooled)

        # Concatenate pooled features from different kernel sizes
        encoding = torch.cat(pooled_outputs, dim=1)  # [batch, filters * len(kernels)]
        return self.dropout(encoding)


class SiameseGatedConvRanker(nn.Module):
    """
    Siamese Ranker using Gated CNNs and Cosine Similarity.
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray = None):
        super(SiameseGatedConvRanker, self).__init__()

        self.embedding_dim = Config.EMBEDDING_DIM

        # Embedding Layer
        self.embedding = nn.Embedding(
            vocab_size, self.embedding_dim, padding_idx=TextUtils.PAD_INDEX
        )

        # Initialize with pre-trained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            # Freeze embeddings as per requirements ("pre-trained static")
            self.embedding.weight.requires_grad = False

        # Shared Encoder
        self.encoder = GatedConvEncoder(
            embedding_dim=self.embedding_dim,
            kernel_sizes=Config.KERNEL_SIZES,
            num_filters=Config.NUM_FILTERS,
            dropout=Config.DROPOUT,
        )

        # Interaction Parameters: Learnable scaling for Cosine Similarity -> Logits
        # Initialize scale to 5.0 to sharpen the cosine distribution for BCE
        self.sim_scale = nn.Parameter(torch.tensor(5.0))
        self.sim_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, q_indices: torch.Tensor, c_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_indices: [batch, q_len]
            c_indices: [batch, c_len] (Training) OR [batch, num_cands, c_len] (Inference)
        Returns:
            logits: [batch] or [batch, num_cands]
        """
        # Embed and Encode Question
        q_emb = self.embedding(q_indices)
        q_vec = self.encoder(q_emb)  # [batch, enc_dim]

        # Handle Inference Mode (Multiple candidates per question)
        if c_indices.dim() == 3:
            batch_size, num_cands, c_len = c_indices.shape

            # Flatten candidates to process in parallel
            c_flat = c_indices.view(-1, c_len)
            c_emb = self.embedding(c_flat)
            c_vec_flat = self.encoder(c_emb)

            # Reshape back to [batch, num_cands, enc_dim]
            enc_dim = c_vec_flat.size(1)
            c_vec = c_vec_flat.view(batch_size, num_cands, enc_dim)

            # Expand Question vector for broadcasting: [batch, 1, enc_dim]
            q_vec_exp = q_vec.unsqueeze(1)

            # Cosine Similarity along the embedding dimension (dim=2)
            sim = F.cosine_similarity(q_vec_exp, c_vec, dim=2)  # [batch, num_cands]

        # Handle Training Mode (Paired candidates, flattened batch)
        else:
            c_emb = self.embedding(c_indices)
            c_vec = self.encoder(c_emb)
            # Cosine Similarity
            sim = F.cosine_similarity(q_vec, c_vec, dim=1)  # [batch]

        # Linear transformation to logits
        logits = sim * self.sim_scale + self.sim_bias
        return logits


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------


def train_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    vocab_size: int,
    embedding_matrix: np.ndarray,
    device: torch.device,
    save_path: str = Config.MODEL_SAVE_PATH,
):
    """
    Trains the Siamese Ranker with Early Stopping.
    """
    model = SiameseGatedConvRanker(vocab_size, embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q_idx = batch["q_indices"].to(device)
            c_idx = batch["c_indices"].to(device)
            labels = batch["labels"].to(device)

            # Flatten batch if collate_fn returned [batch, num_samples, seq_len]
            if c_idx.dim() == 3:
                b, n, l = c_idx.shape
                # Repeat question for each negative sample
                q_idx = q_idx.unsqueeze(1).expand(-1, n, -1).reshape(-1, q_idx.size(1))
                c_idx = c_idx.reshape(-1, l)
                labels = labels.reshape(-1)

            optimizer.zero_grad()
            logits = model(q_idx, c_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                q_idx = batch["q_indices"].to(device)
                c_idx = batch["c_indices"].to(device)
                labels = batch["labels"].to(device)

                # Flatten for validation metric calculation
                if c_idx.dim() == 3:
                    b, n, l = c_idx.shape
                    q_idx = (
                        q_idx.unsqueeze(1).expand(-1, n, -1).reshape(-1, q_idx.size(1))
                    )
                    c_idx = c_idx.reshape(-1, l)
                    labels = labels.reshape(-1)

                logits = model(q_idx, c_idx)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Calculate Accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("  Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("  Early stopping triggered.")
                break

    return model


# -----------------------------------------------------------------------------
# Inference & Heuristics
# -----------------------------------------------------------------------------


def get_short_answer_span(
    question_tokens: List[str], long_answer_text: str, window_size: int = 10
) -> Tuple[int, int, str]:
    """
    Sliding window heuristic to find best short answer span based on n-gram overlap.
    Returns (start_token_offset, end_token_offset, text).
    Offsets are relative to the long_answer_text tokens.
    """
    la_tokens = TextUtils.tokenize(long_answer_text)
    if not la_tokens:
        return -1, -1, ""

    q_set = set(question_tokens)

    max_overlap = -1
    best_window = (-1, -1)

    # Slide window
    for i in range(len(la_tokens)):
        end = min(i + window_size, len(la_tokens))
        window_tokens = la_tokens[i:end]

        # Count unigram overlap
        overlap = sum(1 for t in window_tokens if t in q_set)

        # Simple bigram overlap check
        if len(window_tokens) > 1 and len(question_tokens) > 1:
            # Create sets of bigrams
            q_bigrams = set(zip(question_tokens, question_tokens[1:]))
            w_bigrams = set(zip(window_tokens, window_tokens[1:]))
            overlap += len(q_bigrams.intersection(w_bigrams))

        if overlap > max_overlap:
            max_overlap = overlap
            best_window = (i, end)

    if max_overlap >= Config.SHORT_OVERLAP_THRESHOLD:
        s, e = best_window
        return s, e, " ".join(la_tokens[s:e])

    return -1, -1, ""


def generate_submission(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    vocab: Dict[str, int],
    device: torch.device,
    jsonl_path: str,
    output_path: str,
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()

    # Build File Index for random access to JSONL to retrieve token offsets
    file_index = DataFactory.build_file_index(jsonl_path, load_cached_data=True)

    # Reverse Vocab for heuristic (Index -> Word)
    rev_vocab = {v: k for k, v in vocab.items()}

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            ex_ids = batch["example_ids"]
            q_idx = batch["q_indices"].to(device)
            c_idx = batch["c_indices"].to(device)
            cand_indices = batch["candidate_indices"]  # [batch, max_cands]

            # Compute scores
            logits = model(q_idx, c_idx)  # [batch, max_cands]
            probs = torch.sigmoid(logits)

            # Process each example in the batch
            for i, ex_id in enumerate(ex_ids):
                # Filter out padding candidates (index -1)
                valid_mask = cand_indices[i] != -1

                if not valid_mask.any():
                    # No candidates available
                    results.append(f"{ex_id}_long,")
                    results.append(f"{ex_id}_short,")
                    continue

                valid_probs = probs[i][valid_mask]
                valid_c_indices = cand_indices[i][valid_mask]

                # Find best candidate
                best_score, best_arg = torch.max(valid_probs, dim=0)

                # Check Confidence Threshold
                if best_score.item() < Config.LONG_CONFIDENCE_THRESHOLD:
                    results.append(f"{ex_id}_long,")
                    results.append(f"{ex_id}_short,")
                    continue

                # Retrieve offsets from raw JSON
                # We need the index of the candidate within the specific example's candidate list
                best_cand_idx_in_json = valid_c_indices[best_arg].item()

                offset = file_index[ex_id]
                with open(jsonl_path, "rb") as f:
                    f.seek(offset)
                    entry = json.loads(f.readline())

                c_info = entry["long_answer_candidates"][best_cand_idx_in_json]
                long_start = c_info["start_token"]
                long_end = c_info["end_token"]

                # Add Long Answer Prediction
                results.append(f"{ex_id}_long,{long_start}:{long_end}")

                # Short Answer Logic
                doc_text = entry["document_text"]
                doc_tokens = doc_text.split()

                # Extract long answer text
                ls = max(0, long_start)
                le = min(len(doc_tokens), long_end)
                long_text = " ".join(doc_tokens[ls:le])

                # Reconstruct question text (skip PAD/UNK for cleaner matching)
                q_tokens = [
                    rev_vocab.get(idx.item(), "") for idx in q_idx[i] if idx.item() > 1
                ]

                s_rel_start, s_rel_end, s_text = get_short_answer_span(
                    q_tokens, long_text
                )

                if s_rel_start != -1:
                    # Convert relative offsets to absolute document offsets
                    abs_start = long_start + s_rel_start
                    abs_end = long_start + s_rel_end

                    # Yes/No Check
                    clean_s = s_text.lower().strip()
                    # Simple heuristic for Yes/No
                    if clean_s.startswith("yes") and len(clean_s) < 10:
                        short_ans_str = "YES"
                    elif clean_s.startswith("no") and len(clean_s) < 10:
                        short_ans_str = "NO"
                    else:
                        short_ans_str = f"{abs_start}:{abs_end}"

                    results.append(f"{ex_id}_short,{short_ans_str}")
                else:
                    results.append(f"{ex_id}_short,")

    # Save Submission
    with open(output_path, "w") as f:
        f.write("example_id,PredictionString\n")
        f.write("\n".join(results))
    print(f"Submission saved to {output_path}")
