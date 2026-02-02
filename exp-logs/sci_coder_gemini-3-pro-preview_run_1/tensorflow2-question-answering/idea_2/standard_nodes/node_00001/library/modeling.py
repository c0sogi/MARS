import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from library.config import Config
from library.utils import load_glove_embeddings, tokenize, format_submission
from library.data_loader import (
    get_long_answer_loader,
    process_short_answer_data,
    build_vocab,
)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DEConvNet(nn.Module):
    """
    Dual-Encoder Convolutional Network for Long Answer Ranking.
    """

    def __init__(self, embedding_matrix, freeze_embeddings=False):
        super(DEConvNet, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings,
            padding_idx=0,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=Config.CNN_NUM_FILTERS,
                    kernel_size=k,
                )
                for k in Config.CNN_KERNEL_SIZES
            ]
        )

        # Interaction dimension: (num_filters * num_kernels) * 3 (u, v, u*v)
        self.enc_dim = Config.CNN_NUM_FILTERS * len(Config.CNN_KERNEL_SIZES)
        self.interaction_dim = self.enc_dim * 3

        self.mlp = nn.Sequential(
            nn.Linear(self.interaction_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, 1),
            nn.Sigmoid(),
        )

    def forward_branch(self, x):
        # x: (batch, seq_len)
        # embed: (batch, seq_len, emb_dim) -> (batch, emb_dim, seq_len)
        x = self.embedding(x).permute(0, 2, 1)

        pooled_outputs = []
        for conv in self.convs:
            # conv(x): (batch, filters, L_out)
            # max_pool: (batch, filters)
            c = conv(x)
            # Handle cases where sequence length < kernel size
            if c.size(2) == 0:
                p = torch.zeros(c.size(0), c.size(1), device=c.device)
            else:
                p = torch.max(c, dim=2)[0]
            pooled_outputs.append(p)

        # (batch, total_filters)
        return torch.cat(pooled_outputs, dim=1)

    def forward(self, question, candidate):
        q_vec = self.forward_branch(question)
        c_vec = self.forward_branch(candidate)

        # Interaction: Concatenation + Element-wise Product
        combined = torch.cat([q_vec, c_vec, q_vec * c_vec], dim=1)

        score = self.mlp(combined)
        return score.squeeze(-1)


class WindowLogisticRegressor(nn.Module):
    """
    Lightweight Logistic Regression for Short Answer Extraction.
    """

    def __init__(self, input_dim=4):
        super(WindowLogisticRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, features):
        return self.activation(self.linear(features)).squeeze(-1)


def train_long_answer_model(train_loader, val_loader, model, epochs=Config.NUM_EPOCHS):
    """
    Trains the DEConvNet model with Early Stopping.
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Long Answer Model Training...")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q = batch["question"].to(DEVICE)
            c = batch["candidate"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(q, c)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * q.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                q = batch["question"].to(DEVICE)
                c = batch["candidate"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                preds = model(q, c)
                loss = criterion(preds, labels)
                val_loss += loss.item() * q.size(0)

                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.LONG_ANSWER_MODEL_PATH)
            print("  Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("  Early stopping triggered.")
                break

    # Load best model
    if os.path.exists(Config.LONG_ANSWER_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.LONG_ANSWER_MODEL_PATH, map_location=DEVICE)
        )

    return model


def train_short_answer_model(X, y, epochs=50):
    """
    Trains the WindowLogisticRegressor using PyTorch.
    """
    print("Starting Short Answer Model Training...")

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = WindowLogisticRegressor(input_dim=X.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(X)

        # Simple logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    # Save weights (using numpy for simplicity as it's just a linear layer)
    weights = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()

    np.save(Config.SHORT_ANSWER_WEIGHTS_PATH, {"weights": weights, "bias": bias})
    print(f"Short answer weights saved to {Config.SHORT_ANSWER_WEIGHTS_PATH}")

    return model


def extract_short_answer_inference(
    doc_tokens, q_tokens_set, long_start, long_end, sa_model
):
    """
    Extracts short answer from a long answer span using the trained model.
    """
    la_tokens = doc_tokens[long_start:long_end]
    if not la_tokens:
        return ""

    w_size = Config.WINDOW_SIZE
    stride = Config.WINDOW_STRIDE

    features_list = []
    windows_indices = []

    for i in range(0, len(la_tokens) - w_size + 1, stride):
        window_tokens = la_tokens[i : i + w_size]

        # Features
        has_digit = 1 if any(t.isdigit() for t in window_tokens) else 0
        is_cap = 1 if window_tokens and window_tokens[0][0].isupper() else 0
        match_count = sum(1 for t in window_tokens if t.lower() in q_tokens_set)
        rel_pos = i / len(la_tokens) if len(la_tokens) > 0 else 0

        features_list.append([has_digit, is_cap, match_count, rel_pos])

        # Store absolute token indices
        abs_start = long_start + i
        abs_end = abs_start + w_size
        windows_indices.append((abs_start, abs_end))

    if not features_list:
        return ""

    # Batch prediction
    X_tensor = torch.tensor(features_list, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        scores = sa_model(X_tensor).cpu().numpy()

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score > Config.SHORT_ANSWER_THRESHOLD:
        start, end = windows_indices[best_idx]
        return f"{start}:{end}"

    return ""


def run_inference():
    """
    Runs the full inference pipeline on the test set.
    """
    print("Running Inference...")

    # 1. Load Metadata
    train_meta = pd.read_parquet(Config.TRAIN_META_FILE)
    test_meta = pd.read_parquet(Config.TEST_META_FILE)

    # 2. Build/Load Vocab & Embeddings
    vocab = build_vocab(train_meta, load_cached_data=True)
    embedding_matrix = load_glove_embeddings(
        vocab, Config.EMBEDDING_DIM, load_cached_data=True
    )

    # 3. Load Models
    # Long Answer Model
    la_model = DEConvNet(embedding_matrix).to(DEVICE)
    if os.path.exists(Config.LONG_ANSWER_MODEL_PATH):
        la_model.load_state_dict(
            torch.load(Config.LONG_ANSWER_MODEL_PATH, map_location=DEVICE)
        )
        print("Loaded Long Answer Model.")
    else:
        print(
            "Warning: Long Answer Model not found. Predictions will be random/untrained."
        )
    la_model.eval()

    # Short Answer Model
    sa_model = WindowLogisticRegressor(input_dim=4).to(DEVICE)
    if os.path.exists(Config.SHORT_ANSWER_WEIGHTS_PATH):
        weights_dict = np.load(
            Config.SHORT_ANSWER_WEIGHTS_PATH, allow_pickle=True
        ).item()
        sa_model.linear.weight.data = torch.tensor(weights_dict["weights"]).to(DEVICE)
        sa_model.linear.bias.data = torch.tensor(weights_dict["bias"]).to(DEVICE)
        print("Loaded Short Answer Model.")
    else:
        print("Warning: Short Answer Model not found.")
    sa_model.eval()

    # 4. Long Answer Prediction
    # We use the dataloader to efficiently batch process Q/C pairs
    test_loader = get_long_answer_loader(
        test_meta,
        Config.TEST_DATA_FILE,
        vocab,
        split="test",
        shuffle=False,
        load_cached_data=False,
    )

    # Store best candidate per example
    # example_id -> (best_prob, best_cand_idx)
    best_candidates = {}

    print("Predicting Long Answers...")
    with torch.no_grad():
        for batch in test_loader:
            q = batch["question"].to(DEVICE)
            c = batch["candidate"].to(DEVICE)
            e_ids = batch["example_id"]
            c_idxs = batch["candidate_index"]

            scores = la_model(q, c).cpu().numpy()

            for eid, c_idx, score in zip(e_ids, c_idxs, scores):
                c_idx = int(c_idx)
                if eid not in best_candidates:
                    best_candidates[eid] = (-1.0, -1)

                if score > best_candidates[eid][0]:
                    best_candidates[eid] = (float(score), c_idx)

    # 5. Short Answer Extraction & Final Formatting
    print("Predicting Short Answers and Formatting...")
    final_predictions = {}

    # We need to access text for Short Answer extraction.
    # Iterate test metadata and look up the chosen candidate.

    with open(Config.TEST_DATA_FILE, "rb") as f:
        for _, row in test_meta.iterrows():
            eid = row["example_id"]
            offset = row["byte_offset"]

            if eid not in best_candidates:
                final_predictions[eid] = {"long": "", "short": ""}
                continue

            score, cand_idx = best_candidates[eid]

            # Apply Long Answer Threshold
            if score < Config.LONG_ANSWER_THRESHOLD or cand_idx == -1:
                final_predictions[eid] = {"long": "", "short": ""}
                continue

            # Read data for text processing
            f.seek(offset)
            line = f.readline()
            if not line:
                continue
            data = json.loads(line)

            candidates = data.get("long_answer_candidates", [])
            if cand_idx >= len(candidates):
                final_predictions[eid] = {"long": "", "short": ""}
                continue

            cand = candidates[cand_idx]
            start_token = cand["start_token"]
            end_token = cand["end_token"]

            long_ans_str = f"{start_token}:{end_token}"

            # Short Answer Extraction
            doc_text = data.get("document_text", "")
            doc_tokens = tokenize(doc_text)
            q_text = data.get("question_text", "")
            q_tokens_set = set(tokenize(q_text.lower()))

            short_ans_str = extract_short_answer_inference(
                doc_tokens, q_tokens_set, start_token, end_token, sa_model
            )

            final_predictions[eid] = {"long": long_ans_str, "short": short_ans_str}

    # 6. Save Submission
    format_submission(final_predictions)


def run_training_pipeline():
    """
    Orchestrates the training of both models.
    """
    # 1. Metadata & Vocab
    train_meta = pd.read_parquet(Config.TRAIN_META_FILE)
    val_meta = pd.read_parquet(Config.VAL_META_FILE)

    vocab = build_vocab(train_meta, load_cached_data=True)
    embedding_matrix = load_glove_embeddings(
        vocab, Config.EMBEDDING_DIM, load_cached_data=True
    )

    # 2. Train Long Answer Model
    train_loader = get_long_answer_loader(
        train_meta, Config.TRAIN_DATA_FILE, vocab, split="train", load_cached_data=True
    )
    val_loader = get_long_answer_loader(
        val_meta, Config.TRAIN_DATA_FILE, vocab, split="val", load_cached_data=True
    )

    la_model = DEConvNet(embedding_matrix)
    train_long_answer_model(train_loader, val_loader, la_model)

    # 3. Train Short Answer Model
    X, y = process_short_answer_data(
        train_meta, Config.TRAIN_DATA_FILE, vocab, load_cached_data=True
    )
    train_short_answer_model(X, y)
