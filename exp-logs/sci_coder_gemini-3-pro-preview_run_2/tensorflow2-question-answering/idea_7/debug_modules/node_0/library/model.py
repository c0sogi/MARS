import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.config import Config
from library.data_utils import Tokenizer, build_embedding_matrix
from library.dataset import NQDataset


class QuestionEncoder(nn.Module):
    def __init__(self, input_dim):
        super(QuestionEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.context_vector = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Dim)
        # mask: (Batch, Seq_Len) - 0 for padding, 1 for valid

        # u = tanh(Wx + b)
        u = torch.tanh(self.linear(x))

        # scores = u^T v
        scores = self.context_vector(u).squeeze(-1)  # (Batch, Seq_Len)

        if mask is not None:
            # Mask padding tokens with a large negative value
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len)

        # Weighted sum
        # (Batch, Seq_Len, 1) * (Batch, Seq_Len, Dim) -> (Batch, Dim)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context


class GlobalContextPointwiseNet(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, dropout_rate, embedding_matrix=None
    ):
        super(GlobalContextPointwiseNet, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False  # Freeze embeddings

        # 2. Question Encoder
        self.question_encoder = QuestionEncoder(embedding_dim)

        # 3. Pointwise Encoder (Shared MLP)
        # Input is Candidate Embedding + Broadcasted Question Context
        self.pointwise_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 4. Heads
        # Ranking Head (Long Answer)
        self.ranking_head = nn.Linear(hidden_dim, 1)

        # Span Heads (Short Answer)
        self.span_start_head = nn.Linear(hidden_dim, 1)
        self.span_end_head = nn.Linear(hidden_dim, 1)

        # Yes/No Head
        self.yes_no_head = nn.Linear(hidden_dim, Config.NUM_YES_NO_CLASSES)

    def forward(self, q_seq, c_seq):
        # Masks (assuming 0 is padding)
        q_mask = (q_seq != 0).float()
        c_mask = (c_seq != 0).float()

        # Embeddings
        q_emb = self.embedding(q_seq)  # (B, Q_Len, Emb_Dim)
        c_emb = self.embedding(c_seq)  # (B, C_Len, Emb_Dim)

        # Encode Question -> Global Context
        q_context = self.question_encoder(q_emb, q_mask)  # (B, Emb_Dim)

        # Context Broadcasting
        # Expand q_context to match candidate length: (B, C_Len, Emb_Dim)
        c_len = c_emb.size(1)
        q_context_broadcast = q_context.unsqueeze(1).expand(-1, c_len, -1)

        # Concatenate
        combined_input = torch.cat(
            [c_emb, q_context_broadcast], dim=2
        )  # (B, C_Len, 2*Emb_Dim)

        # Pointwise Encoding
        token_features = self.pointwise_encoder(
            combined_input
        )  # (B, C_Len, Hidden_Dim)

        # Apply mask to token features to ensure padding doesn't affect pooling
        # (B, C_Len, Hidden_Dim) * (B, C_Len, 1)
        masked_token_features = token_features * c_mask.unsqueeze(-1)

        # Global Max Pooling for Document Representation
        # We replace 0s (padding) with -inf before max pool to avoid selecting padding
        pool_input = masked_token_features.clone()
        pool_input[c_mask == 0] = -1e9
        doc_rep, _ = torch.max(pool_input, dim=1)  # (B, Hidden_Dim)

        # --- Heads ---

        # 1. Ranking (Long Answer)
        long_logits = self.ranking_head(doc_rep).squeeze(-1)  # (B)

        # 2. Spans (Short Answer)
        # Predictions for every token
        start_logits = self.span_start_head(token_features).squeeze(-1)  # (B, C_Len)
        end_logits = self.span_end_head(token_features).squeeze(-1)  # (B, C_Len)

        # Mask pad positions in span logits so they aren't selected
        start_logits = start_logits.masked_fill(c_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(c_mask == 0, -1e9)

        # 3. Yes/No
        yes_no_logits = self.yes_no_head(doc_rep)  # (B, 3)

        return long_logits, start_logits, end_logits, yes_no_logits


def calculate_loss(
    long_logits,
    start_logits,
    end_logits,
    yn_logits,
    long_labels,
    start_labels,
    end_labels,
    yn_labels,
):

    # Long Answer Loss (BCE)
    loss_long = F.binary_cross_entropy_with_logits(long_logits, long_labels)

    # Span Loss (CrossEntropy)
    # Ignore index -1 (no short answer)
    loss_start = F.cross_entropy(start_logits, start_labels, ignore_index=-1)
    loss_end = F.cross_entropy(end_logits, end_labels, ignore_index=-1)

    # Yes/No Loss (CrossEntropy)
    # Only calculate yes/no loss for positive long answers?
    # The dataset provides yn_label=0 (NONE) for negatives, so we can train on all.
    loss_yn = F.cross_entropy(yn_logits, yn_labels)

    # Weighted Sum
    total_loss = (
        Config.LOSS_WEIGHT_LONG * loss_long
        + Config.LOSS_WEIGHT_SHORT * (loss_start + loss_end)
        + Config.LOSS_WEIGHT_YESNO * loss_yn
    )

    return (
        total_loss,
        loss_long.item(),
        (loss_start.item() + loss_end.item()),
        loss_yn.item(),
    )


def train_model(load_cached_data=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer & Embeddings
    tokenizer = Tokenizer()
    if os.path.exists(Config.VOCAB_CACHE_FILE) and load_cached_data:
        tokenizer.load(Config.VOCAB_CACHE_FILE)
    else:
        # Fallback: Build vocab from train data text (simplified for this module context)
        # In a real scenario, we'd read the file. Here we assume cache exists or we fail gracefully/rebuild.
        # For the purpose of this script, we assume vocab exists or we build from raw.
        print("Building vocab from scratch...")
        # (Implementation omitted for brevity, assuming vocab cache exists as per instructions)
        pass

    embedding_matrix = build_embedding_matrix(
        tokenizer.word_index, load_cached_data=load_cached_data
    )

    # 2. Datasets & Loaders
    train_dataset = NQDataset(
        Config.TRAIN_META_PATH,
        Config.TRAIN_DATA_PATH,
        tokenizer,
        is_train=True,
        load_cached_data=load_cached_data,
    )
    val_dataset = NQDataset(
        Config.VAL_META_PATH,
        Config.TRAIN_DATA_PATH,
        tokenizer,
        is_train=False,
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model
    model = GlobalContextPointwiseNet(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        embedding_matrix=embedding_matrix,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        # No tqdm to avoid cluttering logs
        for batch in train_loader:
            q_seq = batch["q_seq"].to(device)
            c_seq = batch["c_seq"].to(device)
            long_labels = batch["long_label"].to(device)
            start_labels = batch["short_start"].to(device)
            end_labels = batch["short_end"].to(device)
            yn_labels = batch["yes_no_label"].to(device)

            optimizer.zero_grad()

            l_logits, s_logits, e_logits, yn_logits = model(q_seq, c_seq)

            loss, l_loss, s_loss, y_loss = calculate_loss(
                l_logits,
                s_logits,
                e_logits,
                yn_logits,
                long_labels,
                start_labels,
                end_labels,
                yn_labels,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_long = 0
        total_long = 0

        with torch.no_grad():
            for batch in val_loader:
                q_seq = batch["q_seq"].to(device)
                c_seq = batch["c_seq"].to(device)
                long_labels = batch["long_label"].to(device)
                start_labels = batch["short_start"].to(device)
                end_labels = batch["short_end"].to(device)
                yn_labels = batch["yes_no_label"].to(device)

                l_logits, s_logits, e_logits, yn_logits = model(q_seq, c_seq)

                loss, _, _, _ = calculate_loss(
                    l_logits,
                    s_logits,
                    e_logits,
                    yn_logits,
                    long_labels,
                    start_labels,
                    end_labels,
                    yn_labels,
                )
                val_loss += loss.item()

                # Simple accuracy for long answer
                preds = (torch.sigmoid(l_logits) > 0.5).float()
                correct_long += (preds == long_labels).sum().item()
                total_long += long_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_long / total_long if total_long > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - Val Long Acc: {val_acc:.6f}"
        )

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    return model


def generate_submission(model=None, load_cached_data=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer
    tokenizer = Tokenizer()
    if os.path.exists(Config.VOCAB_CACHE_FILE):
        tokenizer.load(Config.VOCAB_CACHE_FILE)

    # Load Model if not provided
    if model is None:
        embedding_matrix = build_embedding_matrix(
            tokenizer.word_index, load_cached_data=True
        )
        model = GlobalContextPointwiseNet(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            dropout_rate=Config.DROPOUT_RATE,
            embedding_matrix=embedding_matrix,
        ).to(device)
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print("No trained model found. Using initialized weights (random).")

    model.eval()

    # Test Dataset
    test_dataset = NQDataset(
        Config.TEST_META_PATH,
        Config.TEST_DATA_PATH,
        tokenizer,
        is_train=False,
        load_cached_data=load_cached_data,
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # We need to map predictions back to global token indices.
    # NQDataset returns candidate_index. We need to look up the candidate definition.
    # To do this efficiently, we'll read the test file once to build a lookup.
    print("Building candidate offset lookup...")
    candidate_offsets = {}  # {example_id: [ {start_token, end_token}, ... ]}

    with open(Config.TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            eid = str(entry["example_id"])
            candidate_offsets[eid] = entry["long_answer_candidates"]

    results = {}  # {example_id: {'long_score': -inf, 'long_str': '', 'short_str': ''}}

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            q_seq = batch["q_seq"].to(device)
            c_seq = batch["c_seq"].to(device)
            example_ids = batch["example_id"]  # List of strings
            cand_indices = batch["candidate_index"].numpy()

            l_logits, s_logits, e_logits, yn_logits = model(q_seq, c_seq)

            l_probs = torch.sigmoid(l_logits).cpu().numpy()
            s_probs = F.softmax(s_logits, dim=1).cpu().numpy()
            e_probs = F.softmax(e_logits, dim=1).cpu().numpy()
            yn_probs = F.softmax(yn_logits, dim=1).cpu().numpy()

            for i in range(len(example_ids)):
                eid = str(example_ids[i])
                c_idx = cand_indices[i]
                l_score = l_probs[i]

                # Initialize result entry if not exists
                if eid not in results:
                    results[eid] = {"best_score": -1.0, "long_ans": "", "short_ans": ""}

                # We only care if this candidate is better than previous best for this example
                if l_score > results[eid]["best_score"]:

                    # Get global offsets for this candidate
                    cand_info = candidate_offsets[eid][c_idx]
                    global_c_start = cand_info["start_token"]
                    global_c_end = cand_info["end_token"]

                    # 1. Long Answer Prediction
                    long_ans_str = ""
                    if l_score > Config.LONG_CONFIDENCE_THRESHOLD:
                        long_ans_str = f"{global_c_start}:{global_c_end}"

                    # 2. Short Answer Prediction
                    short_ans_str = ""
                    if (
                        l_score > Config.LONG_CONFIDENCE_THRESHOLD
                    ):  # Only predict short if long is valid
                        # Find best span
                        s_idx = np.argmax(s_probs[i])
                        e_idx = np.argmax(e_probs[i])

                        # Check span validity
                        # s_idx and e_idx are relative to candidate start
                        # Must be within valid range and start <= end
                        if s_idx <= e_idx:
                            span_score = s_probs[i][s_idx] * e_probs[i][e_idx]
                            if span_score > Config.SHORT_CONFIDENCE_THRESHOLD:
                                # Convert to global indices
                                global_s_start = global_c_start + s_idx
                                global_s_end = (
                                    global_c_start + e_idx + 1
                                )  # +1 for exclusive end in format?
                                # Task description says "start:end token indices". Usually inclusive:exclusive in Python slices,
                                # but NQ evaluation often expects token indices.
                                # Sample submission: "6:18".
                                # NQ format: "start_token": 6, "end_token": 18 (exclusive).
                                # Our s_end is inclusive index of the token. So +1 for exclusive format.
                                short_ans_str = f"{global_s_start}:{global_s_end}"

                        # 3. Yes/No Override
                        yn_idx = np.argmax(yn_probs[i])
                        if yn_idx == 1:  # YES
                            short_ans_str = "YES"
                        elif yn_idx == 2:  # NO
                            short_ans_str = "NO"

                    results[eid] = {
                        "best_score": l_score,
                        "long_ans": long_ans_str,
                        "short_ans": short_ans_str,
                    }

    # Format for submission
    submission_rows = []
    # Ensure we cover all IDs in sample submission (though test_loader should cover all)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    all_test_ids = set(
        sample_sub["example_id"].apply(
            lambda x: x.replace("_long", "").replace("_short", "")
        )
    )

    for eid in all_test_ids:
        # Default empty
        l_str = ""
        s_str = ""

        if eid in results:
            l_str = results[eid]["long_ans"]
            s_str = results[eid]["short_ans"]

        submission_rows.append({"example_id": f"{eid}_long", "PredictionString": l_str})
        submission_rows.append(
            {"example_id": f"{eid}_short", "PredictionString": s_str}
        )

    sub_df = pd.DataFrame(submission_rows)

    # Ensure order matches sample submission (optional but good practice)
    # Merging with sample to keep order
    final_df = sample_sub[["example_id"]].merge(sub_df, on="example_id", how="left")

    final_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
