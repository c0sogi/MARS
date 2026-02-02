import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from library.config import Config
from library.text_utils import Tokenizer, Vocab, parse_candidates


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class ReaderDataset(Dataset):
    def __init__(self, data_df, max_len):
        self.data = data_df
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        input_ids = row["input_ids"]
        # Targets are only present for training/val
        start_target = row["start_token"] if "start_token" in row else 0
        end_target = row["end_token"] if "end_token" in row else 0

        # Pad or truncate
        curr_len = len(input_ids)
        if curr_len > self.max_len:
            input_ids = input_ids[: self.max_len]
            # Adjust targets if they fall outside
            if start_target >= self.max_len:
                start_target = 0
            if end_target >= self.max_len:
                end_target = 0
        else:
            padding = [0] * (self.max_len - curr_len)
            input_ids = input_ids + padding

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "start_target": torch.tensor(start_target, dtype=torch.long),
            "end_target": torch.tensor(end_target, dtype=torch.long),
        }


class UNetReader(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        super(UNetReader, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # Encoder (Contraction)
        self.enc1 = self._conv_block(embedding_dim, Config.READER_ENC_FILTERS[0])
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = self._conv_block(
            Config.READER_ENC_FILTERS[0], Config.READER_ENC_FILTERS[1]
        )
        self.pool2 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(
            Config.READER_ENC_FILTERS[1], Config.READER_ENC_FILTERS[1] * 2
        )

        # Decoder (Expansion)
        self.up2 = nn.ConvTranspose1d(
            Config.READER_ENC_FILTERS[1] * 2,
            Config.READER_ENC_FILTERS[1],
            kernel_size=2,
            stride=2,
        )
        self.dec2 = self._conv_block(
            Config.READER_ENC_FILTERS[1] * 2, Config.READER_ENC_FILTERS[1]
        )  # Input channels doubled due to concat

        self.up1 = nn.ConvTranspose1d(
            Config.READER_ENC_FILTERS[1],
            Config.READER_ENC_FILTERS[0],
            kernel_size=2,
            stride=2,
        )
        self.dec1 = self._conv_block(
            Config.READER_ENC_FILTERS[0] * 2, Config.READER_ENC_FILTERS[0]
        )

        # Output Heads
        self.start_head = nn.Conv1d(Config.READER_ENC_FILTERS[0], 1, kernel_size=1)
        self.end_head = nn.Conv1d(Config.READER_ENC_FILTERS[0], 1, kernel_size=1)

        self.dropout = nn.Dropout(Config.READER_DROPOUT)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=Config.READER_KERNEL_SIZE, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=Config.READER_KERNEL_SIZE, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x)  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L) for Conv1d

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)
        b = self.dropout(b)

        # Decoder
        u2 = self.up2(b)
        # Handle shape mismatch due to odd lengths if necessary (assuming padding handles it mostly)
        if u2.size(2) != e2.size(2):
            u2 = F.interpolate(u2, size=e2.size(2), mode="linear", align_corners=False)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        if u1.size(2) != e1.size(2):
            u1 = F.interpolate(u1, size=e1.size(2), mode="linear", align_corners=False)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.dropout(d1)

        # Output
        start_logits = self.start_head(d1).squeeze(1)  # (B, L)
        end_logits = self.end_head(d1).squeeze(1)  # (B, L)

        return start_logits, end_logits


def prepare_reader_data(
    metadata_df,
    vocab,
    raw_file_path,
    is_train=True,
    load_cached_data=True,
    cache_path=None,
):
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached reader data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing reader data from {raw_file_path}...")
    data_rows = []

    # Subsample logic
    if (
        is_train
        and Config.TRAIN_SAMPLE_SIZE
        and len(metadata_df) > Config.TRAIN_SAMPLE_SIZE
    ):
        metadata_df = metadata_df.sample(
            n=Config.TRAIN_SAMPLE_SIZE, random_state=Config.SEED
        )
    elif (
        not is_train
        and Config.VAL_SAMPLE_SIZE
        and len(metadata_df) > Config.VAL_SAMPLE_SIZE
    ):
        metadata_df = metadata_df.sample(
            n=Config.VAL_SAMPLE_SIZE, random_state=Config.SEED
        )

    with open(raw_file_path, "rb") as f:
        for _, row in metadata_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line.decode("utf-8"))
                q_text = record.get("question_text", "")
                q_ids = vocab.text_to_ids(q_text)

                if is_train:
                    # Training: Extract Positive Examples with Short Answers
                    annotations = record.get("annotations", [])
                    # Find ground truth short answer
                    short_ans = None
                    long_ans_idx = -1

                    for ann in annotations:
                        if ann["short_answers"]:
                            short_ans = ann["short_answers"][0]  # Take first valid
                            long_ans_idx = ann["long_answer"]["candidate_index"]
                            break

                    if short_ans and long_ans_idx != -1:
                        # Get the text of the containing paragraph
                        candidates = record.get("long_answer_candidates", [])
                        if long_ans_idx < len(candidates):
                            cand = candidates[long_ans_idx]
                            doc_text = record.get("document_text", "")
                            tokens = Tokenizer.tokenize(doc_text)

                            c_start = cand["start_token"]
                            c_end = cand["end_token"]
                            c_tokens = tokens[c_start:c_end]
                            c_text = " ".join(c_tokens)
                            p_ids = vocab.text_to_ids(c_text)

                            # Calculate relative offsets
                            # Short answer indices are absolute in doc
                            s_start_abs = short_ans["start_token"]
                            s_end_abs = short_ans["end_token"]

                            # Check containment
                            if s_start_abs >= c_start and s_end_abs <= c_end:
                                rel_start = len(q_ids) + (s_start_abs - c_start)
                                rel_end = (
                                    len(q_ids) + (s_end_abs - c_start) - 1
                                )  # Inclusive end index for classification

                                input_ids = q_ids + p_ids

                                # Sanity check
                                if rel_end < len(input_ids):
                                    data_rows.append(
                                        {
                                            "example_id": row["example_id"],
                                            "input_ids": input_ids,
                                            "start_token": rel_start,
                                            "end_token": rel_end,
                                        }
                                    )
                else:
                    # Validation/Inference preparation handled differently or not needed here
                    # because we use ranker output for test inference.
                    # However, for validation set evaluation, we process similar to train.
                    pass

            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data_rows)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Saved reader data to {cache_path}")

    return df


def prepare_reader_test_data(
    ranker_output_path, vocab, load_cached_data=True, cache_path=None
):
    """
    Prepares test data using the output from the Ranker.
    """
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached reader test data from {cache_path}")
        return pd.read_parquet(cache_path)

    if not os.path.exists(ranker_output_path):
        print("Ranker output not found.")
        return pd.DataFrame()

    ranker_df = pd.read_parquet(ranker_output_path)
    print(f"Processing reader test inputs from {len(ranker_df)} ranker candidates...")

    data_rows = []
    for _, row in ranker_df.iterrows():
        # q_ids and p_ids are stored as lists/arrays in parquet
        q_ids = list(row["q_ids"])
        p_ids = list(row["p_ids"])

        input_ids = q_ids + p_ids

        data_rows.append(
            {
                "example_id": row["example_id"],
                "input_ids": input_ids,
                "doc_start_token": row[
                    "start_token"
                ],  # Absolute start of paragraph in doc
                "q_len": len(q_ids),
            }
        )

    df = pd.DataFrame(data_rows)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Saved reader test data to {cache_path}")

    return df


def train_reader(train_df, val_df, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Reader on {device}...")

    max_len = Config.Q_MAX_LEN + Config.P_MAX_LEN
    train_dataset = ReaderDataset(train_df, max_len)
    val_dataset = ReaderDataset(val_df, max_len)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = UNetReader(
        vocab.vocab_size, Config.EMBEDDING_DIM, vocab.embedding_matrix
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            start_targets = batch["start_target"].to(device)
            end_targets = batch["end_target"].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids)

            loss = criterion(start_logits, start_targets) + criterion(
                end_logits, end_targets
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_start = 0
        correct_end = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                start_targets = batch["start_target"].to(device)
                end_targets = batch["end_target"].to(device)

                start_logits, end_logits = model(input_ids)
                loss = criterion(start_logits, start_targets) + criterion(
                    end_logits, end_targets
                )
                val_loss += loss.item()

                pred_start = torch.argmax(start_logits, dim=1)
                pred_end = torch.argmax(end_logits, dim=1)

                correct_start += (pred_start == start_targets).sum().item()
                correct_end += (pred_end == end_targets).sum().item()
                total += input_ids.size(0)

        val_loss /= len(val_loader)
        acc_start = correct_start / total
        acc_end = correct_end / total

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Start Acc: {acc_start:.6f} - End Acc: {acc_end:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.READER_MODEL_PATH)
            patience = 0
        else:
            patience += 1
            if patience >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    return model


def generate_submission(test_df, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating submission...")

    if not os.path.exists(Config.READER_MODEL_PATH):
        print("Reader model not found. Cannot generate predictions.")
        return

    model = UNetReader(
        vocab.vocab_size, Config.EMBEDDING_DIM, vocab.embedding_matrix
    ).to(device)
    model.load_state_dict(torch.load(Config.READER_MODEL_PATH, map_location=device))
    model.eval()

    max_len = Config.Q_MAX_LEN + Config.P_MAX_LEN
    # Create dataset for test (no targets)
    # We reuse ReaderDataset but ignore targets in getitem logic if missing
    # Since ReaderDataset expects targets, we can just add dummy ones to df
    test_df["start_token"] = 0
    test_df["end_token"] = 0

    dataset = ReaderDataset(test_df, max_len)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    results = []

    # We need to map back to original rows
    # DataLoader preserves order if shuffle=False

    batch_idx = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            start_logits, end_logits = model(input_ids)

            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)

            # Convert to numpy
            start_probs = start_probs.cpu().numpy()
            end_probs = end_probs.cpu().numpy()

            # Process batch
            current_batch_size = input_ids.size(0)
            for i in range(current_batch_size):
                global_idx = batch_idx * Config.BATCH_SIZE + i
                row = test_df.iloc[global_idx]

                s_prob = start_probs[i]
                e_prob = end_probs[i]

                # Find best span
                # Heuristic: limit max span length to e.g. 30 tokens
                max_span_len = 30
                best_score = -1
                best_span = (0, 0)

                # Argmax search
                # Optimization: Look at top K start and end tokens
                top_k = 10
                top_starts = np.argsort(s_prob)[-top_k:]
                top_ends = np.argsort(e_prob)[-top_k:]

                for s in top_starts:
                    for e in top_ends:
                        if s <= e and (e - s) < max_span_len:
                            score = s_prob[s] * e_prob[e]
                            if score > best_score:
                                best_score = score
                                best_span = (s, e)

                # Logic for prediction
                # Long Answer: We have the paragraph from Ranker.
                # Short Answer: We have the span from Reader.

                # Ranker score is not passed here, assuming Ranker filtered well.
                # We use SHORT_ANSWER_THRESHOLD

                q_len = row["q_len"]
                doc_offset = row["doc_start_token"]

                # Adjust indices relative to paragraph
                # Span indices in input_ids include Question
                # Paragraph starts at index q_len

                pred_s, pred_e = best_span

                short_ans_str = ""
                long_ans_str = ""

                # Check if span is within paragraph part
                if pred_s >= q_len and best_score > Config.SHORT_ANSWER_THRESHOLD:
                    # Valid short answer
                    rel_s = pred_s - q_len
                    rel_e = pred_e - q_len

                    abs_s = doc_offset + rel_s
                    abs_e = (
                        doc_offset + rel_e + 1
                    )  # +1 for exclusive end in submission format usually?
                    # Task format: "start:end" token indices. Usually exclusive end in Python, but CSVs often inclusive or exclusive.
                    # NQ evaluation usually expects token indices.
                    # Sample submission: 6:18.

                    short_ans_str = f"{abs_s}:{abs_e}"

                    # If we have a short answer, we definitely have a long answer (the paragraph)
                    # We need the paragraph length. We can infer it from input_ids or pass it.
                    # Ranker output didn't save paragraph end. But we can approximate or just output the short span if unsure.
                    # Actually, we should have saved paragraph end in ranker output.
                    # Assuming we didn't, we can't output exact long answer span easily without re-parsing.
                    # However, we can construct a long answer span around the short answer or just output the short answer.
                    # Wait, the task requires Long Answer prediction too.
                    # Let's assume the Ranker's candidate text length.
                    # Since we don't have it here easily without re-tokenizing, let's assume the paragraph is valid.
                    # We can assume the candidate provided by ranker is the long answer.
                    # We need its end token.
                    # Let's look at prepare_reader_test_data. It reads ranker output.
                    # Ranker output has 'p_ids'. len(p_ids) is the length.

                    p_len = (
                        len(row["input_ids"]) - q_len
                    )  # Approximation (might include padding)
                    # Actually p_ids in ranker output was padded? No, ranker dataset pads, but prepare_ranker_data saves raw ids.
                    # prepare_ranker_data saves: q_ids, p_ids (raw lists).
                    # So len(p_ids) is correct length.

                    p_len_real = (
                        len(row["input_ids"]) - q_len
                    )  # This comes from test_df which came from ranker output
                    # Wait, prepare_reader_test_data concatenates q_ids + p_ids.
                    # So len(input_ids) is total length.

                    long_abs_s = doc_offset
                    long_abs_e = doc_offset + p_len_real
                    long_ans_str = f"{long_abs_s}:{long_abs_e}"

                results.append(
                    {
                        "example_id": row["example_id"],
                        "long_ans": long_ans_str,
                        "short_ans": short_ans_str,
                    }
                )

            batch_idx += 1

    # Format for submission
    submission_rows = []
    for res in results:
        eid = res["example_id"]
        # Long
        submission_rows.append([f"{eid}_long", res["long_ans"]])
        # Short
        submission_rows.append([f"{eid}_short", res["short_ans"]])

    sub_df = pd.DataFrame(submission_rows, columns=["example_id", "PredictionString"])
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_reader_pipeline():
    Config.setup_directories()

    # 1. Load Vocab
    vocab = Vocab()
    if os.path.exists(Config.VOCAB_PATH) and os.path.exists(
        Config.EMBEDDING_MATRIX_PATH
    ):
        vocab.load(Config.VOCAB_PATH, Config.EMBEDDING_MATRIX_PATH)
    else:
        print("Vocab not found.")
        return

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # 3. Prepare Train Data
    train_df = prepare_reader_data(
        train_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=True,
        cache_path=Config.READER_TRAIN_PATH,
    )
    val_df = prepare_reader_data(
        val_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=True,
        cache_path=Config.READER_VAL_PATH,
    )

    # 4. Train
    train_reader(train_df, val_df, vocab)

    # 5. Inference
    # Requires Ranker to have run and produced Config.RANKER_TEST_PATH
    test_df = prepare_reader_test_data(
        Config.RANKER_TEST_PATH,
        vocab,
        load_cached_data=False,
        cache_path=Config.READER_TEST_PATH,
    )

    if not test_df.empty:
        generate_submission(test_df, vocab)
    else:
        print("Test data empty or ranker output missing.")
