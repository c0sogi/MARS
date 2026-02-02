import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from library.config import Config


class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.itos = {0: Config.PAD_TOKEN, 1: Config.UNK_TOKEN}
        self.stoi = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
        self.freqs = Counter()

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """
        Builds the vocabulary from a list of sentences.
        """
        for sentence in sentence_list:
            if not isinstance(sentence, str):
                continue
            tokens = sentence.lower().split()
            self.freqs.update(tokens)

        idx = 2
        # Sort to ensure deterministic ordering for same frequencies
        for word, freq in sorted(self.freqs.items()):
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        """
        Converts a text string into a list of indices.
        """
        if not isinstance(text, str):
            return [self.stoi[Config.UNK_TOKEN]]

        tokens = text.lower().split()
        result = []
        for token in tokens:
            if token in self.stoi:
                result.append(self.stoi[token])
            else:
                result.append(self.stoi[Config.UNK_TOKEN])
        return result

    def save(self, path):
        """
        Saves the vocabulary mapping to a parquet file.
        """
        data = {"token": list(self.stoi.keys()), "index": list(self.stoi.values())}
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        """
        Loads the vocabulary mapping from a parquet file.
        """
        df = pd.read_parquet(path)
        self.stoi = dict(zip(df["token"], df["index"]))
        self.itos = {i: t for t, i in self.stoi.items()}


class PhraseDataset(Dataset):
    def __init__(self, df, vocab, context_map):
        self.df = df
        self.vocab = vocab
        self.context_map = context_map

        # Pre-fetch columns to lists for faster indexing
        self.ids = df["id"].tolist()
        self.anchors = df["anchor"].tolist()
        self.targets = df["target"].tolist()
        self.contexts = df["context"].tolist()

        if "score" in df.columns:
            self.scores = df["score"].astype(float).tolist()
        else:
            self.scores = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_text = self.anchors[idx]
        target_text = self.targets[idx]
        context_text = self.contexts[idx]

        # Tokenize and Numericalize
        anchor_indices = self.vocab.numericalize(anchor_text)
        target_indices = self.vocab.numericalize(target_text)

        # Truncate to MAX_LEN
        if len(anchor_indices) > Config.MAX_LEN:
            anchor_indices = anchor_indices[: Config.MAX_LEN]
        if len(target_indices) > Config.MAX_LEN:
            target_indices = target_indices[: Config.MAX_LEN]

        # Context Mapping (default to 0 if unseen, though unlikely in this dataset)
        context_idx = self.context_map.get(context_text, 0)

        item = {
            "anchor": torch.tensor(anchor_indices, dtype=torch.long),
            "target": torch.tensor(target_indices, dtype=torch.long),
            "context": torch.tensor(context_idx, dtype=torch.long),
            "id": self.ids[idx],
        }

        if self.scores is not None:
            item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


def collate_fn(batch):
    """
    Custom collate function to pad sequences and stack tensors.
    """
    anchors = [item["anchor"] for item in batch]
    targets = [item["target"] for item in batch]
    contexts = torch.tensor([item["context"] for item in batch], dtype=torch.long)
    ids = [item["id"] for item in batch]

    # Pad sequences with 0 (PAD_TOKEN)
    anchors_padded = pad_sequence(anchors, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    batch_out = {
        "anchor": anchors_padded,
        "target": targets_padded,
        "context": contexts,
        "id": ids,
    }

    if "score" in batch[0]:
        scores = torch.tensor([item["score"] for item in batch], dtype=torch.float)
        batch_out["score"] = scores

    return batch_out


def build_embedding_matrix(vocab):
    """
    Builds a pre-trained embedding matrix using SentenceTransformer.
    Cite solution_lesson_node_00001: Initialize with pre-trained static vectors.
    """
    print("Building pre-trained embedding matrix using SentenceTransformer...")
    try:
        from sentence_transformers import SentenceTransformer

        # Use a small, efficient model to get static embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Extract words from vocab (ordered by index)
        # Index 0 is PAD, 1 is UNK
        words = [vocab.itos[i] for i in range(len(vocab))]

        # Encode words to get their vectors
        embeddings = model.encode(words, show_progress_bar=True, convert_to_numpy=True)

        # Explicitly set PAD token (index 0) to zeros
        embeddings[0] = np.zeros(embeddings.shape[1])

        return embeddings
    except Exception as e:
        print(f"Failed to load SentenceTransformer: {e}. Returning None.")
        return None


def get_dataloaders(load_cached_data=True):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    Handles caching of Vocabulary and Context mappings.

    Args:
        load_cached_data (bool): If True, attempts to load vocab/context maps from disk.
                                 If False or load fails, rebuilds them.

    Returns:
        train_loader, val_loader, test_loader, vocab_size, num_contexts, embedding_matrix
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load DataFrames
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG Mode: Reduced train size to {len(df_train)}")

    # 2. Manage Vocabulary and Context Maps
    vocab_path = os.path.join(Config.WORKING_DIR, "vocab.parquet")
    context_path = os.path.join(Config.WORKING_DIR, "context_map.parquet")
    embed_path = os.path.join(Config.WORKING_DIR, "embeddings.npy")

    vocab = Vocabulary(min_freq=Config.VOCAB_MIN_FREQ)
    context_map = {}
    embedding_matrix = None

    cache_loaded = False

    if load_cached_data:
        if os.path.exists(vocab_path) and os.path.exists(context_path):
            try:
                print("Loading cached vocabulary and context map...")
                vocab.load(vocab_path)

                context_df = pd.read_parquet(context_path)
                context_map = dict(zip(context_df["context"], context_df["index"]))

                # Try to load cached embeddings
                if os.path.exists(embed_path):
                    print("Loading cached embedding matrix...")
                    embedding_matrix = np.load(embed_path)

                cache_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")
        else:
            print("Cache not found. Rebuilding...")

    if not cache_loaded:
        print("Building vocabulary and context map...")
        # Build Vocab from Train (Anchor + Target)
        text_corpus = (
            pd.concat([df_train["anchor"], df_train["target"]]).astype(str).tolist()
        )
        vocab.build_vocabulary(text_corpus)
        vocab.save(vocab_path)

        # Build Context Map
        unique_contexts = sorted(df_train["context"].unique().tolist())
        context_map = {ctx: i for i, ctx in enumerate(unique_contexts)}

        # Save Context Map
        c_df = pd.DataFrame(
            {"context": list(context_map.keys()), "index": list(context_map.values())}
        )
        c_df.to_parquet(context_path, index=False)

    # Build or Rebuild Embeddings if missing
    if embedding_matrix is None:
        embedding_matrix = build_embedding_matrix(vocab)
        if embedding_matrix is not None:
            np.save(embed_path, embedding_matrix)

    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Number of Contexts: {len(context_map)}")
    if embedding_matrix is not None:
        print(f"Embedding Matrix Shape: {embedding_matrix.shape}")

    # 3. Create Datasets
    train_dataset = PhraseDataset(df_train, vocab, context_map)
    val_dataset = PhraseDataset(df_val, vocab, context_map)
    test_dataset = PhraseDataset(df_test, vocab, context_map)

    # 4. Create DataLoaders
    # Use pin_memory=True if GPU is available for faster transfer
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(vocab),
        len(context_map),
        embedding_matrix,
    )
