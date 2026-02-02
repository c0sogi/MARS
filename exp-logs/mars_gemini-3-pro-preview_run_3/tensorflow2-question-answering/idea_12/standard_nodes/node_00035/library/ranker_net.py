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


class RankerDataset(Dataset):
    def __init__(self, data_df, max_q_len, max_p_len):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing 'q_ids', 'p_ids', and 'label'.
            max_q_len (int): Maximum question length.
            max_p_len (int): Maximum paragraph length.
        """
        self.data = data_df
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Cite debug_lesson_4: Explicitly cast to list to handle potential NumPy array
        # representation from Parquet/Pandas loading, preventing broadcast errors during concatenation.
        q_ids = list(row["q_ids"])
        p_ids = list(row["p_ids"])
        label = row["label"] if "label" in row else 0.0

        # Pad or truncate Question
        if len(q_ids) > self.max_q_len:
            q_ids = q_ids[: self.max_q_len]
        else:
            q_ids = q_ids + [0] * (self.max_q_len - len(q_ids))

        # Pad or truncate Paragraph
        if len(p_ids) > self.max_p_len:
            p_ids = p_ids[: self.max_p_len]
        else:
            p_ids = p_ids + [0] * (self.max_p_len - len(p_ids))

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "p_ids": torch.tensor(p_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


class InteractionRanker(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        super(InteractionRanker, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Fine-tuning embeddings is allowed

        # Convolutional Layers
        self.convs = nn.ModuleList()
        in_channels = 1

        for out_channels in Config.RANKER_CONV_FILTERS:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=Config.RANKER_KERNEL_SIZES[0],
                    padding=1,
                )
            )
            in_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=Config.RANKER_POOL_SIZES[0])
        self.dropout = nn.Dropout(Config.RANKER_DROPOUT)

        # Calculate flattened dimension dynamically
        # Assuming input dims are reduced by factor of 2 per pooling layer
        h_q = Config.Q_MAX_LEN
        h_p = Config.P_MAX_LEN
        for _ in Config.RANKER_CONV_FILTERS:
            h_q = h_q // 2
            h_p = h_p // 2

        flat_dim = Config.RANKER_CONV_FILTERS[-1] * h_q * h_p

        self.fc1 = nn.Linear(flat_dim, Config.RANKER_HIDDEN_DIM)
        self.fc2 = nn.Linear(Config.RANKER_HIDDEN_DIM, 1)

    def forward(self, q_ids, p_ids):
        # Embeddings: (B, L, D)
        q_emb = self.embedding(q_ids)
        p_emb = self.embedding(p_ids)

        # Interaction Matrix: (B, Lq, Lp)
        # Compute outer product / similarity
        interaction = torch.bmm(q_emb, p_emb.transpose(1, 2))

        # Add channel dim: (B, 1, Lq, Lp)
        x = interaction.unsqueeze(1)

        # CNN Blocks
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)
            x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return torch.sigmoid(logits).squeeze(1)


def prepare_ranker_data(
    metadata_df,
    vocab,
    raw_file_path,
    is_train=True,
    load_cached_data=True,
    cache_path=None,
):
    """
    Prepares data for the ranker.
    """
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached ranker data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing ranker data from {raw_file_path}...")
    data_rows = []

    # Subsample metadata if configured
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
        # For test/val inference, we might also subsample if configured for debugging
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

                # Parse Question
                q_text = record.get("question_text", "")
                q_ids_raw = vocab.text_to_ids(q_text)

                # Parse Candidates
                candidates = parse_candidates(
                    record.get("document_text", ""),
                    record.get("long_answer_candidates", []),
                    max_candidates=Config.MAX_CANDIDATES_PER_DOC,
                )

                if is_train:
                    # Identify Ground Truth
                    annotations = record.get("annotations", [])
                    gt_indices = set()
                    for ann in annotations:
                        la = ann.get("long_answer", {})
                        if la.get("start_token", -1) != -1:
                            gt_indices.add((la["start_token"], la["end_token"]))

                    pos_candidates = []
                    neg_candidates = []

                    for cand in candidates:
                        key = (cand["start_token"], cand["end_token"])
                        c_ids = vocab.text_to_ids(cand["text"])

                        if key in gt_indices:
                            pos_candidates.append(c_ids)
                        else:
                            neg_candidates.append(c_ids)

                    # Create Pairs (Positive)
                    for p_ids in pos_candidates:
                        data_rows.append(
                            {
                                "example_id": row["example_id"],
                                "q_ids": q_ids_raw,
                                "p_ids": p_ids,
                                "label": 1,
                            }
                        )

                        # Add Negatives (Sampled)
                        if neg_candidates:
                            n_sample = min(len(neg_candidates), Config.NEG_RATIO)
                            sampled_negs = random.sample(neg_candidates, n_sample)
                            for n_ids in sampled_negs:
                                data_rows.append(
                                    {
                                        "example_id": row["example_id"],
                                        "q_ids": q_ids_raw,
                                        "p_ids": n_ids,
                                        "label": 0,
                                    }
                                )
                else:
                    # Inference Mode: Store all candidates
                    for cand in candidates:
                        c_ids = vocab.text_to_ids(cand["text"])
                        data_rows.append(
                            {
                                "example_id": row["example_id"],
                                "q_ids": q_ids_raw,
                                "p_ids": c_ids,
                                "label": 0,  # Dummy
                                "candidate_text": cand["text"],
                                "start_token": cand["start_token"],
                                "end_token": cand["end_token"],
                            }
                        )

            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data_rows)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Saved ranker data to {cache_path}")

    return df


def train_ranker(train_df, val_df, vocab):
    """
    Trains the interaction ranker.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Ranker on {device}...")

    train_dataset = RankerDataset(train_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)
    val_dataset = RankerDataset(val_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = InteractionRanker(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_embeddings=vocab.embedding_matrix,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q_ids = batch["q_ids"].to(device)
            p_ids = batch["p_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(q_ids, p_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * q_ids.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["q_ids"].to(device)
                p_ids = batch["p_ids"].to(device)
                labels = batch["label"].to(device)

                outputs = model(q_ids, p_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * q_ids.size(0)

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)

        preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_labels, preds_binary)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {acc:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return model


def predict_test_candidates(test_df, vocab):
    """
    Runs inference on test candidates and selects the best candidate per example_id.
    Saves the result to Config.RANKER_TEST_PATH for the Reader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Ranker Inference on Test Set...")

    if not os.path.exists(Config.RANKER_MODEL_PATH):
        print("Model file not found. Skipping inference.")
        return

    model = InteractionRanker(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_embeddings=vocab.embedding_matrix,
    ).to(device)
    model.load_state_dict(torch.load(Config.RANKER_MODEL_PATH, map_location=device))
    model.eval()

    dataset = RankerDataset(test_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)
    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=0
    )

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            q_ids = batch["q_ids"].to(device)
            p_ids = batch["p_ids"].to(device)
            outputs = model(q_ids, p_ids)
            all_scores.extend(outputs.cpu().tolist())

    test_df["rank_score"] = all_scores

    # Select best candidate per example_id
    best_candidates = test_df.loc[
        test_df.groupby("example_id")["rank_score"].idxmax()
    ].reset_index(drop=True)

    print(f"Selected {len(best_candidates)} best candidates for reading.")

    # Save for Reader
    best_candidates.to_parquet(Config.RANKER_TEST_PATH, index=False)
    print(f"Saved ranker test features to {Config.RANKER_TEST_PATH}")


def run_ranker_pipeline():
    Config.setup_directories()

    # 1. Load Vocab
    vocab = Vocab()
    if os.path.exists(Config.VOCAB_PATH) and os.path.exists(
        Config.EMBEDDING_MATRIX_PATH
    ):
        vocab.load(Config.VOCAB_PATH, Config.EMBEDDING_MATRIX_PATH)
    else:
        print("Vocab not found. Please ensure vocab is built.")
        return

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Prepare Data
    train_df = prepare_ranker_data(
        train_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=True,
        cache_path=Config.RANKER_TRAIN_PATH,
    )
    val_df = prepare_ranker_data(
        val_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=True,
        cache_path=Config.RANKER_VAL_PATH,
    )

    # 4. Train
    train_ranker(train_df, val_df, vocab)

    # 5. Inference
    test_candidates_df = prepare_ranker_data(
        test_meta,
        vocab,
        Config.TEST_RAW_FILE,
        is_train=False,
        load_cached_data=False,
        cache_path=None,
    )

    predict_test_candidates(test_candidates_df, vocab)
