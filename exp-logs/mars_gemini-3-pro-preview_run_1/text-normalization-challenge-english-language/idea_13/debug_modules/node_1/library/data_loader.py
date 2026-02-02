import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config
from library.utils import set_seed
from library.features import RegexFeaturizer, BPETokenizerWrapper, GlobalPriorMap

# Constants for special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


class Vocabulary:
    def __init__(self, name, special_tokens=None):
        self.name = name
        self.special_tokens = (
            special_tokens if special_tokens else [PAD_TOKEN, UNK_TOKEN]
        )
        self.token2id = {}
        self.id2token = {}

        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token

    def build(self, tokens, min_freq=1, max_size=None):
        counter = Counter(tokens)

        # Sort by frequency then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        curr_idx = len(self.token2id)
        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_size and len(self.token2id) >= max_size:
                break

            if token not in self.token2id:
                self.token2id[token] = curr_idx
                self.id2token[curr_idx] = token
                curr_idx += 1

    def __len__(self):
        return len(self.token2id)

    def to_id(self, token):
        return self.token2id.get(str(token), self.token2id.get(UNK_TOKEN))

    def to_token(self, idx):
        return self.id2token.get(idx, UNK_TOKEN)

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(
                {"token2id": self.token2id, "special_tokens": self.special_tokens}, f
            )

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.token2id = data["token2id"]
            self.special_tokens = data["special_tokens"]
            self.id2token = {int(v): k for k, v in self.token2id.items()}


class KnowledgeBase:
    def __init__(self):
        self.lookup_table = {}  # (token, class) -> normalized

    def build(self, train_csv_path, load_cached_data=True):
        if load_cached_data and os.path.exists(Config.KNOWLEDGE_BASE_PATH):
            print("Loading Knowledge Base from cache...")
            df = pd.read_parquet(Config.KNOWLEDGE_BASE_PATH)
            # Create a multi-index map
            self.lookup_table = df.set_index(["before", "class"])["after"].to_dict()
            return

        print("Building Knowledge Base from scratch...")
        df = pd.read_csv(
            train_csv_path,
            usecols=["before", "class", "after"],
            dtype=str,
            keep_default_na=False,
        )

        # Find most frequent normalization for each (token, class) pair
        # We use mode; if multiple modes, pick the first one (alphabetically or by appearance)
        # Using groupby and agg with a lambda for mode is slow.
        # Faster: value_counts -> sort -> drop_duplicates

        counts = (
            df.groupby(["before", "class", "after"]).size().reset_index(name="count")
        )
        counts = counts.sort_values(["count", "after"], ascending=[False, True])
        best_normalization = counts.drop_duplicates(
            subset=["before", "class"], keep="first"
        )

        # Save to parquet
        best_normalization[["before", "class", "after"]].to_parquet(
            Config.KNOWLEDGE_BASE_PATH, index=False
        )

        self.lookup_table = best_normalization.set_index(["before", "class"])[
            "after"
        ].to_dict()
        print(f"Knowledge Base built with {len(self.lookup_table)} entries.")

    def get(self, token, token_class):
        return self.lookup_table.get((str(token), str(token_class)), None)


class TaggerDataset(Dataset):
    def __init__(self, split_name, features_dict, metadata_df):
        """
        Args:
            split_name: 'train', 'val', or 'test'
            features_dict: Dictionary containing numpy arrays for features
            metadata_df: DataFrame containing sentence_id, token_id info
        """
        self.split = split_name
        self.word_ids = features_dict["word_ids"]
        self.char_ids = features_dict["char_ids"]
        self.bpe_ids = features_dict["bpe_ids"]
        self.regex_feats = features_dict["regex_feats"]
        self.prior_feats = features_dict["prior_feats"]
        self.labels = features_dict.get("labels", None)
        self.metadata = metadata_df

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        item = {
            "word_ids": torch.tensor(self.word_ids[idx], dtype=torch.long),
            "char_ids": torch.tensor(self.char_ids[idx], dtype=torch.long),
            "bpe_ids": torch.tensor(self.bpe_ids[idx], dtype=torch.long),
            "regex_feats": torch.tensor(self.regex_feats[idx], dtype=torch.float32),
            "prior_feats": torch.tensor(self.prior_feats[idx], dtype=torch.float32),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        # We don't return metadata in the tensor dict to facilitate default collation
        # Metadata can be accessed via index if needed outside
        return item


class FallbackDataset(Dataset):
    def __init__(self, data_df, char_vocab, class_vocab):
        self.data = data_df
        self.char_vocab = char_vocab
        self.class_vocab = class_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = row["before"]
        tgt_text = row["after"]
        cls_name = row["class"]

        # Tokenize
        src_ids = [self.char_vocab.to_id(c) for c in src_text]
        tgt_ids = [self.char_vocab.to_id(c) for c in tgt_text] + [
            self.char_vocab.to_id(EOS_TOKEN)
        ]
        cls_id = self.class_vocab.to_id(cls_name)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "class_id": torch.tensor(cls_id, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        # Dynamic padding
        src_ids = [item["src_ids"] for item in batch]
        tgt_ids = [item["tgt_ids"] for item in batch]
        class_ids = torch.stack([item["class_id"] for item in batch])

        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_ids, batch_first=True, padding_value=0
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_ids, batch_first=True, padding_value=0
        )

        return {"src_ids": src_padded, "tgt_ids": tgt_padded, "class_id": class_ids}


def build_vocabularies(train_csv_path, load_cached_data=True):
    # Paths
    word_vocab_path = Config.VOCAB_WORDS_PATH
    char_vocab_path = Config.VOCAB_CHARS_PATH
    class_vocab_path = Config.VOCAB_CLASSES_PATH

    word_vocab = Vocabulary("words")
    char_vocab = Vocabulary(
        "chars", special_tokens=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    )
    class_vocab = Vocabulary(
        "classes", special_tokens=[]
    )  # Classes usually don't need PAD/UNK if we know all classes

    if (
        load_cached_data
        and os.path.exists(word_vocab_path)
        and os.path.exists(char_vocab_path)
        and os.path.exists(class_vocab_path)
    ):
        print("Loading vocabularies from cache...")
        word_vocab.load(word_vocab_path)
        char_vocab.load(char_vocab_path)
        class_vocab.load(class_vocab_path)
        return word_vocab, char_vocab, class_vocab

    print("Building vocabularies from scratch...")
    df = pd.read_csv(train_csv_path, dtype=str, keep_default_na=False)

    # Words
    word_vocab.build(
        df["before"].tolist(),
        min_freq=Config.MIN_FREQ,
        max_size=Config.VOCAB_SIZE_WORDS,
    )
    word_vocab.save(word_vocab_path)

    # Chars (from both before and after)
    all_chars = set()
    for text in df["before"].dropna():
        all_chars.update(str(text))
    for text in df["after"].dropna():
        all_chars.update(str(text))
    char_vocab.build(list(all_chars))
    char_vocab.save(char_vocab_path)

    # Classes
    class_vocab.build(df["class"].dropna().tolist())
    class_vocab.save(class_vocab_path)

    return word_vocab, char_vocab, class_vocab


def process_tagger_data(
    csv_path, split_name, word_vocab, char_vocab, class_vocab, load_cached_data=True
):
    """
    Processes data for the Bi-LSTM Tagger.
    Groups by sentence_id, pads to MAX_SEQ_LEN, and saves as .npy files.
    """
    cache_prefix = os.path.join(Config.WORKING_DIR, f"{split_name}_tagger")

    # Check if all files exist
    required_files = [
        "word_ids.npy",
        "char_ids.npy",
        "bpe_ids.npy",
        "regex_feats.npy",
        "prior_feats.npy",
        "metadata.parquet",
    ]
    if split_name != "test":
        required_files.append("labels.npy")

    all_exist = all(os.path.exists(f"{cache_prefix}_{f}") for f in required_files)

    if load_cached_data and all_exist:
        print(f"Loading cached tagger data for {split_name}...")
        features = {}
        features["word_ids"] = np.load(f"{cache_prefix}_word_ids.npy")
        features["char_ids"] = np.load(f"{cache_prefix}_char_ids.npy")
        features["bpe_ids"] = np.load(f"{cache_prefix}_bpe_ids.npy")
        features["regex_feats"] = np.load(f"{cache_prefix}_regex_feats.npy")
        features["prior_feats"] = np.load(f"{cache_prefix}_prior_feats.npy")
        metadata = pd.read_parquet(f"{cache_prefix}_metadata.parquet")

        if split_name != "test":
            features["labels"] = np.load(f"{cache_prefix}_labels.npy")

        return features, metadata

    print(f"Processing tagger data for {split_name}...")

    # Load Data
    cols = ["sentence_id", "token_id", "before", "id"]
    if split_name != "test":
        cols.extend(["class"])

    df = pd.read_csv(csv_path, usecols=cols, dtype=str, keep_default_na=False)

    # Convert sentence_id to int for sorting
    df["sentence_id"] = df["sentence_id"].astype(int)
    df["token_id"] = df["token_id"].astype(int)
    df = df.sort_values(["sentence_id", "token_id"])

    # Initialize Featurizers
    regex_featurizer = RegexFeaturizer()
    bpe_tokenizer = BPETokenizerWrapper()
    bpe_tokenizer.load()
    prior_map = GlobalPriorMap()
    prior_map.build(
        Config.TRAIN_DATA, load_cached_data=True
    )  # Always build/load priors from train

    # Group by sentence
    grouped = df.groupby("sentence_id")

    # Prepare lists to collect processed data
    batch_word_ids = []
    batch_char_ids = []
    batch_bpe_ids = []
    batch_regex = []
    batch_priors = []
    batch_labels = []

    # Metadata collection
    meta_sentence_ids = []
    meta_token_ids = []  # List of lists (as strings for storage)
    meta_ids = []

    # Constants
    MAX_SEQ = Config.MAX_SEQ_LEN
    MAX_WORD_LEN = Config.MAX_WORD_LEN
    MAX_BPE_LEN = 10  # Fixed subword length

    # Iterate sentences
    # To optimize, we can process in chunks, but for clarity/robustness we iterate
    # Given 220GB RAM, we can build lists and convert to numpy at the end

    for sent_id, group in grouped:
        tokens = group["before"].tolist()

        # Truncate if too long
        if len(tokens) > MAX_SEQ:
            tokens = tokens[:MAX_SEQ]
            group = group.iloc[:MAX_SEQ]

        curr_len = len(tokens)

        # Word IDs
        w_ids = [word_vocab.to_id(t) for t in tokens]
        # Pad
        w_ids += [0] * (MAX_SEQ - curr_len)
        batch_word_ids.append(w_ids)

        # Char IDs (Seq, Word_Len)
        c_ids_sent = np.zeros((MAX_SEQ, MAX_WORD_LEN), dtype=np.int32)
        for i, t in enumerate(tokens):
            chars = [char_vocab.to_id(c) for c in str(t)][:MAX_WORD_LEN]
            c_ids_sent[i, : len(chars)] = chars
        batch_char_ids.append(c_ids_sent)

        # BPE IDs (Seq, BPE_Len)
        # Process batch of tokens for this sentence
        bpe_tensor = bpe_tokenizer.encode_as_padded_tensor(tokens, max_len=MAX_BPE_LEN)
        # Pad sentence dim
        bpe_padded = np.zeros((MAX_SEQ, MAX_BPE_LEN), dtype=np.int32)
        bpe_padded[:curr_len, :] = bpe_tensor
        batch_bpe_ids.append(bpe_padded)

        # Regex Features (Seq, Dim)
        reg_tensor = regex_featurizer.transform(tokens)
        reg_padded = np.zeros((MAX_SEQ, Config.REGEX_DIM), dtype=np.float32)
        reg_padded[:curr_len, :] = reg_tensor
        batch_regex.append(reg_padded)

        # Prior Features (Seq, Dim)
        prior_tensor = prior_map.get_priors(tokens)
        prior_padded = np.zeros((MAX_SEQ, Config.PRIOR_DIM), dtype=np.float32)
        prior_padded[:curr_len, :] = prior_tensor
        batch_priors.append(prior_padded)

        # Labels
        if split_name != "test":
            classes = group["class"].tolist()
            l_ids = [class_vocab.to_id(c) for c in classes]
            # Pad labels with -100 (standard ignore index) or 0?
            # We'll use 0 (assuming 0 is PAD/Plain or handled by mask) or a specific ignore val.
            # Let's use 0 for now, masking should be handled by length.
            # Actually, standard PyTorch CrossEntropy ignores -100.
            l_ids += [-100] * (MAX_SEQ - curr_len)
            batch_labels.append(l_ids)

        # Metadata
        # We store metadata as parallel arrays (N_sentences, MAX_SEQ)
        # But strings in numpy are messy. We'll store a flattened dataframe in parquet
        # that maps (row_idx, col_idx) -> id.
        # Actually, simpler: Store a DataFrame with one row per sentence containing lists of IDs.
        meta_sentence_ids.append(sent_id)
        meta_token_ids.append(group["token_id"].tolist())
        meta_ids.append(group["id"].tolist())

    # Convert to Numpy
    print("Converting to numpy arrays...")
    feat_dict = {
        "word_ids": np.array(batch_word_ids, dtype=np.int32),
        "char_ids": np.array(batch_char_ids, dtype=np.int32),
        "bpe_ids": np.array(batch_bpe_ids, dtype=np.int32),
        "regex_feats": np.array(batch_regex, dtype=np.float32),
        "prior_feats": np.array(batch_priors, dtype=np.float32),
    }

    if split_name != "test":
        feat_dict["labels"] = np.array(batch_labels, dtype=np.int32)

    # Save Metadata
    # We create a dataframe where each row is a sentence
    meta_df = pd.DataFrame(
        {
            "sentence_id": meta_sentence_ids,
            "token_ids": [
                json.dumps(x) for x in meta_token_ids
            ],  # Serialize list to string
            "ids": [json.dumps(x) for x in meta_ids],
        }
    )

    # Save to Cache
    print("Saving to cache...")
    np.save(f"{cache_prefix}_word_ids.npy", feat_dict["word_ids"])
    np.save(f"{cache_prefix}_char_ids.npy", feat_dict["char_ids"])
    np.save(f"{cache_prefix}_bpe_ids.npy", feat_dict["bpe_ids"])
    np.save(f"{cache_prefix}_regex_feats.npy", feat_dict["regex_feats"])
    np.save(f"{cache_prefix}_prior_feats.npy", feat_dict["prior_feats"])
    meta_df.to_parquet(f"{cache_prefix}_metadata.parquet", index=False)

    if split_name != "test":
        np.save(f"{cache_prefix}_labels.npy", feat_dict["labels"])

    return feat_dict, meta_df


def get_tagger_loader(split_name, batch_size=32, shuffle=True, load_cached_data=True):
    """
    Returns a DataLoader for the Tagger model.
    """
    # Determine path
    if split_name == "train":
        csv_path = Config.TRAIN_DATA
    elif split_name == "val":
        csv_path = Config.VAL_DATA
    else:
        csv_path = Config.TEST_DATA

    # Load Vocabs
    word_vocab, char_vocab, class_vocab = build_vocabularies(
        Config.TRAIN_DATA, load_cached_data
    )

    # Process Data
    features, metadata = process_tagger_data(
        csv_path, split_name, word_vocab, char_vocab, class_vocab, load_cached_data
    )

    # Create Dataset
    dataset = TaggerDataset(split_name, features, metadata)

    # Create Loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader, (word_vocab, char_vocab, class_vocab)


def get_fallback_loader(split_name, batch_size=32, shuffle=True, load_cached_data=True):
    """
    Returns a DataLoader for the Fallback Seq2Seq model.
    Only includes 'changed' tokens from train/val.
    """
    if split_name == "test":
        # Fallback loader for test isn't used directly for training.
        # Inference uses the model directly on OOV tokens.
        return None, None

    cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_fallback.parquet")

    # Load Vocabs
    _, char_vocab, class_vocab = build_vocabularies(Config.TRAIN_DATA, load_cached_data)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached fallback data for {split_name}...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"Processing fallback data for {split_name}...")
        # Load Raw
        csv_path = Config.TRAIN_DATA if split_name == "train" else Config.VAL_DATA
        df = pd.read_csv(
            csv_path,
            usecols=["before", "after", "class"],
            dtype=str,
            keep_default_na=False,
        )

        # Filter changed
        df = df[df["before"] != df["after"]].copy()

        # Save
        df.to_parquet(cache_path, index=False)
        print(f"Saved {len(df)} changed examples to {cache_path}")

    dataset = FallbackDataset(df, char_vocab, class_vocab)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=FallbackDataset.collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return loader
