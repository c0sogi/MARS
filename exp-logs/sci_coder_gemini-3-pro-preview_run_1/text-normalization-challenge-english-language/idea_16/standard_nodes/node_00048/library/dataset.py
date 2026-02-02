import os
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.features import process_dataset

# =========================================================================
# Vocabulary Management
# =========================================================================


class Vocab:
    """Generic Vocabulary for mapping tokens to IDs."""

    def __init__(self, name, specials=None):
        self.name = name
        self.token2id = {}
        self.id2token = {}
        self.specials = specials if specials is not None else ["<PAD>", "<UNK>"]

        for i, token in enumerate(self.specials):
            self.token2id[token] = i
            self.id2token[i] = token

    def __len__(self):
        return len(self.token2id)

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

    def lookup(self, token):
        res = self.token2id.get(token)
        if res is not None:
            return res
        res = self.token2id.get("<UNK>")
        if res is not None:
            return res
        raise ValueError(
            f"Token '{token}' not found in vocab '{self.name}' and no <UNK> token defined."
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}


def build_artifacts(load_cached_data=True):
    """
    Builds and saves:
    1. Word Vocabulary (for Tagger)
    2. Class Vocabulary (for Tagger targets)
    3. Seq2Seq Character Vocabulary (for Fallback)
    4. Knowledge Base (for deterministic lookup)
    """
    word_vocab_path = os.path.join(Config.VOCAB_DIR, "vocab_words.json")
    class_vocab_path = os.path.join(Config.VOCAB_DIR, "vocab_classes.json")
    seq2seq_vocab_path = os.path.join(Config.VOCAB_DIR, "vocab_seq2seq.json")
    kb_path = os.path.join(Config.CACHE_DIR, "knowledge_base.parquet")

    if load_cached_data and os.path.exists(word_vocab_path) and os.path.exists(kb_path):
        print("Loading artifacts from cache...")
        return

    print("Building artifacts from training data...")
    df = pd.read_csv(Config.TRAIN_DATA_PATH, keep_default_na=False)

    # 1. Word Vocab
    print("Building Word Vocab...")
    word_counts = Counter(df["before"].astype(str))
    common_words = [
        w
        for w, c in word_counts.most_common(Config.MAX_VOCAB_SIZE_WORD)
        if c >= Config.MIN_FREQ_WORD
    ]
    word_vocab = Vocab("words")
    word_vocab.add_tokens(common_words)
    word_vocab.save(word_vocab_path)

    # 2. Class Vocab
    print("Building Class Vocab...")
    classes = sorted(df["class"].unique().astype(str))
    class_vocab = Vocab(
        "classes", specials=[]
    )  # No padding needed for class labels usually, but handled by collate
    class_vocab.add_tokens(classes)
    class_vocab.save(class_vocab_path)

    # 3. Seq2Seq Char Vocab
    print("Building Seq2Seq Char Vocab...")
    # Include both source and target characters
    chars = set()
    # Filter for changed tokens to focus vocab on relevant transformations
    changed_df = df[df["before"] != df["after"]]
    for text in changed_df["before"].astype(str):
        chars.update(text)
    for text in changed_df["after"].astype(str):
        chars.update(text)

    char_vocab = Vocab("chars", specials=["<PAD>", "<UNK>", "<SOS>", "<EOS>"])
    char_vocab.add_tokens(sorted(list(chars)))
    char_vocab.save(seq2seq_vocab_path)

    # 4. Knowledge Base
    print("Building Knowledge Base...")
    # Map (token, class) -> after
    # We take the most frequent normalization for a pair if conflicts exist (rare)
    kb_df = (
        df.groupby(["before", "class"])["after"]
        .agg(lambda x: pd.Series.mode(x)[0])
        .reset_index()
    )
    kb_df.to_parquet(kb_path, index=False)
    print("Artifacts built and saved.")


# =========================================================================
# Density Sampling
# =========================================================================


def perform_density_sampling(df):
    """
    Filters the dataframe to:
    1. Keep 100% of sentences containing at least one rare class (non-PLAIN, non-PUNCT).
    2. Subsample sentences containing ONLY PLAIN/PUNCT classes.
    """
    print(f"Performing Density Sampling (Plain Keep Rate: {Config.PLAIN_KEEP_RATE})...")

    # Identify sentence types
    # 0 = Boring (Plain/Punct only), 1 = Interesting
    df["is_interesting"] = ~df["class"].isin(["PLAIN", "PUNCT"])

    # Group by sentence to find max 'is_interesting' (if any token is interesting, sentence is interesting)
    sent_interest = df.groupby("sentence_id")["is_interesting"].max()

    interesting_sents = sent_interest[sent_interest].index
    boring_sents = sent_interest[~sent_interest].index

    # Sample boring sentences
    # Set seed for reproducibility
    rng = np.random.RandomState(Config.SEED)
    keep_boring = rng.choice(
        boring_sents,
        size=int(len(boring_sents) * Config.PLAIN_KEEP_RATE),
        replace=False,
    )

    valid_sents = set(interesting_sents) | set(keep_boring)

    # Filter original DF
    df_sampled = df[df["sentence_id"].isin(valid_sents)].copy()

    print(f"Original Sentences: {len(sent_interest)}")
    print(
        f"Selected Sentences: {len(valid_sents)} ({len(interesting_sents)} interesting + {len(keep_boring)} sampled)"
    )
    print(f"Original Tokens: {len(df)}")
    print(f"Selected Tokens: {len(df_sampled)}")

    return df_sampled


# =========================================================================
# Datasets
# =========================================================================


class TaggerDataset(Dataset):
    def __init__(self, split, load_cached_data=True):
        self.split = split

        # 1. Load Data
        csv_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        self.df = pd.read_csv(csv_path, keep_default_na=False)

        # Store original index to map back to pre-computed features
        self.df["original_index"] = np.arange(len(self.df))

        # 2. Density Sampling (Train only)
        if split == "train":
            self.df = perform_density_sampling(self.df)

        # 3. Sort for Sentence Grouping
        # Ensure data is sorted by sentence_id and token_id
        self.df.sort_values(["sentence_id", "token_id"], inplace=True)

        # 4. Load Features (Pre-computed for the FULL split)
        # process_dataset returns features aligned with the raw CSV
        feats = process_dataset(split, load_cached_data=load_cached_data)
        self.char_features_all = feats["char_features"]
        self.regex_features_all = feats["regex_features"]

        # 5. Load Vocabs
        self.word_vocab = Vocab("words")
        self.word_vocab.load(os.path.join(Config.VOCAB_DIR, "vocab_words.json"))

        if split != "test":
            self.class_vocab = Vocab("classes", specials=[])
            self.class_vocab.load(os.path.join(Config.VOCAB_DIR, "vocab_classes.json"))

        # 6. Pre-compute Word IDs
        print("Mapping words to IDs...")
        self.word_ids = np.array(
            [self.word_vocab.lookup(w) for w in self.df["before"].astype(str)]
        )

        # 7. Pre-compute Labels
        if split != "test":
            print("Mapping classes to IDs...")
            self.labels = np.array(
                [self.class_vocab.lookup(c) for c in self.df["class"].astype(str)]
            )
        else:
            self.labels = None

        # 8. Group Indices by Sentence
        # We use numpy to find split points for speed
        print("Grouping by sentence...")
        sentence_ids = self.df["sentence_id"].values
        unique_sents, split_indices = np.unique(sentence_ids, return_index=True)
        # split_indices gives the start index of each unique sentence in the SORTED dataframe
        # We need to construct (start, end) pairs
        self.sent_starts = split_indices
        self.sent_ends = np.append(split_indices[1:], len(self.df))
        self.num_sentences = len(unique_sents)

        # Store the 'original_index' column as array for fast lookup
        self.original_indices = self.df["original_index"].values

    def __len__(self):
        return self.num_sentences

    def __getitem__(self, idx):
        # Get start and end pointers for the sentence in the dataframe
        start = self.sent_starts[idx]
        end = self.sent_ends[idx]

        # Get the original indices to fetch external features
        orig_idxs = self.original_indices[start:end]

        # Fetch data
        # Truncate to MAX_SEQ_LEN if necessary (though rare for sentences)
        length = min(end - start, Config.MAX_SEQ_LEN)

        # Slicing
        # Note: We take the first 'length' tokens if sentence is too long
        current_word_ids = self.word_ids[start : start + length]

        # Features from the global arrays using original indices
        # We must use the specific original indices corresponding to these rows
        current_orig_idxs = orig_idxs[:length]
        current_char_feats = self.char_features_all[current_orig_idxs]
        current_regex_feats = self.regex_features_all[current_orig_idxs]

        item = {
            "word_ids": torch.tensor(current_word_ids, dtype=torch.long),
            "char_features": torch.tensor(current_char_feats, dtype=torch.long),
            "regex_features": torch.tensor(current_regex_feats, dtype=torch.float32),
            "original_indices": torch.tensor(
                current_orig_idxs, dtype=torch.long
            ),  # Useful for tracking
        }

        if self.labels is not None:
            current_labels = self.labels[start : start + length]
            item["labels"] = torch.tensor(current_labels, dtype=torch.long)

        return item

    @staticmethod
    def collate_fn(batch):
        # Pad sequences in the batch
        # word_ids: [Batch, Seq]
        word_ids = pad_sequence(
            [b["word_ids"] for b in batch], batch_first=True, padding_value=0
        )

        # regex_features: [Batch, Seq, FeatDim]
        regex_features = pad_sequence(
            [b["regex_features"] for b in batch], batch_first=True, padding_value=0
        )

        # char_features: [Batch, Seq, CharLen]
        # This is 3D padding. pad_sequence works on the first dim (Seq).
        # We need to manually handle the 3rd dim or just use pad_sequence if the inner tensors are fixed size?
        # The inner tensors are [Seq, CharLen]. CharLen is fixed (MAX_CHAR_LEN).
        # So pad_sequence will result in [Batch, MaxSeq, CharLen]. Correct.
        char_features = pad_sequence(
            [b["char_features"] for b in batch], batch_first=True, padding_value=0
        )

        batch_out = {
            "word_ids": word_ids,
            "char_features": char_features,
            "regex_features": regex_features,
            "lengths": torch.tensor(
                [len(b["word_ids"]) for b in batch], dtype=torch.long
            ),
        }

        if "labels" in batch[0]:
            labels = pad_sequence(
                [b["labels"] for b in batch], batch_first=True, padding_value=-100
            )  # -100 for ignore_index
            batch_out["labels"] = labels

        # Keep track of IDs for submission mapping
        if "original_indices" in batch[0]:
            # We can't easily pad indices, but usually we just need them for inference order.
            # For inference, batch size 1 or carefully handling order is preferred.
            # We'll just return a list of lists for metadata
            batch_out["original_indices"] = [b["original_indices"] for b in batch]

        return batch_out


class Seq2SeqDataset(Dataset):
    def __init__(self, split, load_cached_data=True):
        self.split = split

        # 1. Load Data
        csv_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        df = pd.read_csv(csv_path, keep_default_na=False)

        # 2. Filter (Train/Val only)
        # We only train on tokens that actually change or are complex
        if split != "test":
            self.df = df[df["before"] != df["after"]].copy()
            # Filter out empty inputs which cause pack_padded_sequence to fail
            # Also ensure target is not empty/null
            self.df = self.df[
                (self.df["before"].astype(str).str.len() > 0)
                & (self.df["after"].astype(str).str.len() > 0)
            ].copy()
        else:
            # For test, we might technically use this dataset for inference on OOV items
            # But usually inference is done via the pipeline.
            # If we need a dataset for test, it would be all tokens.
            self.df = df.copy()

        # 3. Load Vocabs
        self.char_vocab = Vocab("chars")
        self.char_vocab.load(os.path.join(Config.VOCAB_DIR, "vocab_seq2seq.json"))

        self.class_vocab = Vocab("classes", specials=[])
        self.class_vocab.load(os.path.join(Config.VOCAB_DIR, "vocab_classes.json"))

        # 4. Pre-process
        self.inputs = self.df["before"].astype(str).tolist()

        if split != "test":
            self.targets = self.df["after"].astype(str).tolist()
            self.classes = self.df["class"].astype(str).tolist()
        else:
            self.targets = None
            # Test set doesn't have classes provided, they come from Tagger predictions.
            # This Dataset is primarily for TRAINING the fallback model.
            # Inference uses the model directly on strings.
            self.classes = None

    def __len__(self):
        return len(self.inputs)

    def text_to_indices(self, text, add_special=False):
        indices = [self.char_vocab.lookup(c) for c in text]
        if add_special:
            indices = (
                [self.char_vocab.lookup("<SOS>")]
                + indices
                + [self.char_vocab.lookup("<EOS>")]
            )
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        src_text = self.inputs[idx]
        src_ids = self.text_to_indices(
            src_text, add_special=False
        )  # Encoder usually doesn't need SOS/EOS, but depends on impl

        item = {"src_ids": src_ids, "raw_before": src_text}

        if self.targets is not None:
            tgt_text = self.targets[idx]
            tgt_ids = self.text_to_indices(
                tgt_text, add_special=True
            )  # Decoder needs SOS/EOS
            item["tgt_ids"] = tgt_ids
            item["raw_after"] = tgt_text

            # Add class conditioning
            class_name = self.classes[idx]
            # Handle OOV classes safely (though rare)
            try:
                class_id = self.class_vocab.lookup(class_name)
            except ValueError:
                # Fallback to a default class (e.g., 0) if OOV
                class_id = 0

            item["class_id"] = torch.tensor(class_id, dtype=torch.long)

        return item

    @staticmethod
    def collate_fn(batch):
        src_ids = pad_sequence(
            [b["src_ids"] for b in batch], batch_first=True, padding_value=0
        )

        batch_out = {
            "src_ids": src_ids,
            "src_lens": torch.tensor(
                [len(b["src_ids"]) for b in batch], dtype=torch.long
            ),
        }

        if "tgt_ids" in batch[0]:
            tgt_ids = pad_sequence(
                [b["tgt_ids"] for b in batch], batch_first=True, padding_value=0
            )
            batch_out["tgt_ids"] = tgt_ids
            batch_out["tgt_lens"] = torch.tensor(
                [len(b["tgt_ids"]) for b in batch], dtype=torch.long
            )
            batch_out["class_ids"] = torch.stack([b["class_id"] for b in batch])

        return batch_out


# =========================================================================
# Factory Functions
# =========================================================================


def get_tagger_loader(split, batch_size=Config.BATCH_SIZE, shuffle=True):
    dataset = TaggerDataset(split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TaggerDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )


def get_seq2seq_loader(split, batch_size=Config.BATCH_SIZE, shuffle=True):
    dataset = Seq2SeqDataset(split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Seq2SeqDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )
