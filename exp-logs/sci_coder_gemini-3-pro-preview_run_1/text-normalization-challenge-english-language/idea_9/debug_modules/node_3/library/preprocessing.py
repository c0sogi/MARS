import os
import pandas as pd
import numpy as np
import sentencepiece as spm
from collections import Counter
import torch
from library.config import Config
from library.utils import set_seed


# ==========================================
# Vocabulary Management
# ==========================================
class Vocabulary:
    def __init__(self, specials=None):
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else []

    def build(self, tokens, max_size=None, min_freq=1):
        """Build vocabulary from a list of tokens."""
        counter = Counter(tokens)

        # Sort by frequency then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        # Start with specials
        self.stoi = {tok: i for i, tok in enumerate(self.specials)}
        self.itos = {i: tok for i, tok in enumerate(self.specials)}
        idx = len(self.specials)

        for tok, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_size and idx >= max_size:
                break
            self.stoi[tok] = idx
            self.itos[idx] = tok
            idx += 1

    def __len__(self):
        return len(self.stoi)

    def lookup_indices(self, tokens, unk_token=None):
        unk_idx = self.stoi.get(unk_token) if unk_token else None
        return [self.stoi.get(tok, unk_idx) for tok in tokens]

    def lookup_token(self, idx):
        return self.itos.get(idx, None)

    def save(self, path):
        """Save vocabulary to parquet."""
        data = [{"token": tok, "index": idx} for tok, idx in self.stoi.items()]
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        """Load vocabulary from parquet."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        df = pd.read_parquet(path)
        self.stoi = dict(zip(df["token"], df["index"]))
        self.itos = dict(zip(df["index"], df["token"]))


# ==========================================
# BPE Tokenizer Wrapper
# ==========================================
class BPETokenizer:
    def __init__(self, model_prefix, vocab_size):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        self.model_path = f"{model_prefix}.model"

    def train(self, text_file):
        """Train SentencePiece model."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_prefix), exist_ok=True)

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=Config.PAD_IDX,
            unk_id=Config.UNK_IDX,
            bos_id=Config.SOS_IDX,
            eos_id=Config.EOS_IDX,
            pad_piece=Config.PAD_TOKEN,
            unk_piece=Config.UNK_TOKEN,
            bos_piece=Config.SOS_TOKEN,
            eos_piece=Config.EOS_TOKEN,
            user_defined_symbols=[],
        )
        self.sp.load(self.model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BPE model not found at {self.model_path}")
        self.sp.load(self.model_path)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)


# ==========================================
# Knowledge Base
# ==========================================
class KnowledgeBase:
    def __init__(self):
        self.kb = {}  # (before, class) -> after

    def build(self, df):
        """
        Build KB from dataframe.
        Expects columns: 'before', 'class', 'after'.
        """
        # We only care about consistent mappings.
        # If there are conflicts, we take the most frequent one (mode).
        # However, for this task, usually (token, class) -> normalized is deterministic.
        # We'll assume the last seen or simple iteration is sufficient,
        # but to be robust let's drop duplicates.

        # Filter relevant columns
        subset = df[["before", "class", "after"]].dropna()

        # Convert to dictionary
        # Iterating might be slow, let's use pandas logic
        # Drop duplicates to keep unique mappings
        unique_mappings = subset.drop_duplicates(subset=["before", "class"])

        for _, row in unique_mappings.iterrows():
            key = (str(row["before"]), str(row["class"]))
            self.kb[key] = str(row["after"])

    def get(self, before, class_name):
        return self.kb.get((str(before), str(class_name)))

    def save(self, path):
        """Save KB to parquet."""
        data = []
        for (before, cls), after in self.kb.items():
            data.append({"before": before, "class": cls, "after": after})
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        """Load KB from parquet."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"KB file not found at {path}")
        df = pd.read_parquet(path)
        # Reconstruct dictionary
        self.kb = {}
        for _, row in df.iterrows():
            self.kb[(row["before"], row["class"])] = row["after"]


# ==========================================
# Pipeline Functions
# ==========================================


def build_vocabularies(train_df, load_cached=True):
    """
    Builds or loads Word, Char, Class vocabularies and BPE model.
    """
    set_seed(Config.SEED)

    # Paths
    word_path = Config.VOCAB_WORDS_PATH
    char_path = Config.VOCAB_CHARS_PATH
    class_path = Config.VOCAB_CLASSES_PATH
    bpe_prefix = Config.VOCAB_BPE_MODEL_PATH
    bpe_model_path = f"{bpe_prefix}.model"

    # Initialize
    word_vocab = Vocabulary(specials=[Config.PAD_TOKEN, Config.UNK_TOKEN])
    char_vocab = Vocabulary(
        specials=[
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
        ]
    )
    class_vocab = Vocabulary(
        specials=[]
    )  # No specials for classes usually, or maybe PAD if we pad labels
    # We pad labels for batching, so let's add PAD to class vocab or handle it separately.
    # Usually CrossEntropyLoss ignores index -100 or we use 0 as PAD.
    # Let's add PAD to class vocab to be safe and use 0 as PAD.
    class_vocab = Vocabulary(specials=[Config.PAD_TOKEN])

    bpe_tokenizer = BPETokenizer(bpe_prefix, Config.BPE_VOCAB_SIZE)

    # Check cache
    all_exist = (
        os.path.exists(word_path)
        and os.path.exists(char_path)
        and os.path.exists(class_path)
        and os.path.exists(bpe_model_path)
    )

    if load_cached and all_exist:
        print("Loading vocabularies from cache...")
        word_vocab.load(word_path)
        char_vocab.load(char_path)
        class_vocab.load(class_path)
        bpe_tokenizer.load()
        return word_vocab, char_vocab, class_vocab, bpe_tokenizer

    print("Building vocabularies from scratch...")

    # Ensure string types
    tokens = train_df["before"].astype(str).tolist()
    classes = train_df["class"].astype(str).tolist()
    targets = train_df["after"].astype(str).tolist()

    # 1. Word Vocab
    word_vocab.build(tokens, max_size=Config.WORD_VOCAB_SIZE)
    word_vocab.save(word_path)

    # 2. Char Vocab (from before and after)
    all_chars = set()
    for t in tokens:
        all_chars.update(t)
    for t in targets:
        all_chars.update(t)
    char_vocab.build(list(all_chars), max_size=Config.CHAR_VOCAB_SIZE)
    char_vocab.save(char_path)

    # 3. Class Vocab
    class_vocab.build(classes)
    class_vocab.save(class_path)

    # 4. BPE Tokenizer
    # Create temp file for training
    temp_text_file = os.path.join(Config.WORKING_DIR, "bpe_train_corpus.txt")
    with open(temp_text_file, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")

    bpe_tokenizer.train(temp_text_file)

    # Cleanup
    if os.path.exists(temp_text_file):
        os.remove(temp_text_file)

    return word_vocab, char_vocab, class_vocab, bpe_tokenizer


def build_knowledge_base(train_df, load_cached=True):
    """
    Builds or loads the deterministic Knowledge Base.
    """
    kb_path = Config.KNOWLEDGE_BASE_PATH
    kb = KnowledgeBase()

    if load_cached and os.path.exists(kb_path):
        print("Loading Knowledge Base from cache...")
        kb.load(kb_path)
        return kb

    print("Building Knowledge Base...")
    kb.build(train_df)
    kb.save(kb_path)
    return kb


def process_tagger_data(
    df,
    word_vocab,
    char_vocab,
    class_vocab,
    bpe_tokenizer,
    mode="train",
    load_cached=True,
):
    """
    Prepares data for the Tagger (Bi-LSTM).
    Groups tokens by sentence_id.
    Returns a pandas DataFrame where each row is a sentence with list-columns.
    """
    set_seed(Config.SEED)

    # Determine output path based on mode
    if mode == "train":
        out_path = Config.TRAIN_TAGGER_DATA_PATH
    elif mode == "val":
        out_path = Config.VAL_TAGGER_DATA_PATH
    else:
        out_path = Config.TEST_TAGGER_DATA_PATH

    if load_cached and os.path.exists(out_path):
        print(f"Loading {mode} tagger data from {out_path}...")
        return pd.read_parquet(out_path)

    print(f"Processing {mode} tagger data...")

    # Ensure types
    df = df.copy()
    df["before"] = df["before"].astype(str)
    if "class" in df.columns:
        df["class"] = df["class"].astype(str)

    # 1. Word IDs
    # Use apply for speed on large series
    unk_word_idx = word_vocab.stoi[Config.UNK_TOKEN]
    df["word_id"] = df["before"].apply(lambda x: word_vocab.stoi.get(x, unk_word_idx))

    # 2. Char IDs (List of ints)
    unk_char_idx = char_vocab.stoi[Config.UNK_TOKEN]

    def encode_chars(text):
        # Truncate to MAX_CHAR_LEN
        chars = list(text)[: Config.MAX_CHAR_LEN]
        return [char_vocab.stoi.get(c, unk_char_idx) for c in chars]

    df["char_ids"] = df["before"].apply(encode_chars)

    # 3. BPE IDs (List of ints)
    # BPE encode returns list of ints
    df["bpe_ids"] = df["before"].apply(lambda x: bpe_tokenizer.encode(x))

    # 4. Label IDs
    if mode != "test" and "class" in df.columns:
        # Assuming class vocab has PAD at 0, we just map directly.
        # Unknown classes shouldn't theoretically exist in val/test if built from train,
        # but safe to handle.
        df["label_id"] = df["class"].apply(
            lambda x: class_vocab.stoi.get(x, 0)
        )  # Default to PAD/0 if unknown? Or create UNK class?
        # Ideally, all classes are known. If not, 0 (PAD) is effectively 'ignore' in loss usually.
    else:
        df["label_id"] = 0

    # 5. Group by Sentence ID
    # We need to aggregate into lists.
    # Sort first to ensure token order
    df = df.sort_values(["sentence_id", "token_id"])

    # Group
    # Note: 'id' column is unique per token, we don't need to aggregate it for training,
    # but for submission (test) we might need to reconstruct.
    # For Tagger training, we just need features and labels.
    # For Test, we need to know which token is which.
    # Let's aggregate 'id' as well.

    agg_dict = {
        "word_id": list,
        "char_ids": list,
        "bpe_ids": list,
        "label_id": list,
        "id": list,
    }

    grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

    # Save
    grouped.to_parquet(out_path, index=False)

    return grouped


def process_seq2seq_data(df, char_vocab, class_vocab, mode="train", load_cached=True):
    """
    Prepares data for the Seq2Seq Fallback Model.
    Filters for changed tokens (before != after).
    Returns a DataFrame with src_ids, class_id, tgt_ids.
    """
    set_seed(Config.SEED)

    if mode == "train":
        out_path = Config.TRAIN_SEQ2SEQ_DATA_PATH
    elif mode == "val":
        out_path = Config.VAL_SEQ2SEQ_DATA_PATH
    else:
        # We don't pre-process test data for seq2seq in bulk usually,
        # because seq2seq is run conditionally based on Tagger output.
        # But if we wanted to, we could. Here we focus on train/val.
        return None

    if load_cached and os.path.exists(out_path):
        print(f"Loading {mode} seq2seq data from {out_path}...")
        return pd.read_parquet(out_path)

    print(f"Processing {mode} seq2seq data...")

    df = df.copy()
    df["before"] = df["before"].astype(str)
    df["after"] = df["after"].astype(str)
    df["class"] = df["class"].astype(str)

    # Filter for changes
    # We also want to train on some identity mappings?
    # The prompt says "specialized dataset by filtering... where before != after".
    # This focuses the model on transformations.
    changed_df = df[df["before"] != df["after"]].copy()

    if len(changed_df) == 0:
        print("Warning: No changed tokens found for Seq2Seq data.")
        return pd.DataFrame()

    unk_char_idx = char_vocab.stoi[Config.UNK_TOKEN]
    sos_idx = char_vocab.stoi[Config.SOS_TOKEN]
    eos_idx = char_vocab.stoi[Config.EOS_TOKEN]

    # Encode Source (before)
    # Just chars. No SOS/EOS needed for Encoder usually, but depends on implementation.
    # Standard Transformer Encoder takes raw sequence.
    def encode_src(text):
        chars = list(text)[: Config.MAX_SEQ2SEQ_LEN]
        return [char_vocab.stoi.get(c, unk_char_idx) for c in chars]

    # Encode Target (after)
    # Needs SOS and EOS for Decoder training
    def encode_tgt(text):
        chars = list(text)[: (Config.MAX_SEQ2SEQ_LEN - 2)]  # Reserve space
        ids = [char_vocab.stoi.get(c, unk_char_idx) for c in chars]
        return [sos_idx] + ids + [eos_idx]

    changed_df["src_ids"] = changed_df["before"].apply(encode_src)
    changed_df["tgt_ids"] = changed_df["after"].apply(encode_tgt)

    # Class ID
    changed_df["class_id"] = changed_df["class"].apply(
        lambda x: class_vocab.stoi.get(x, 0)
    )

    # Select columns
    out_df = changed_df[["src_ids", "class_id", "tgt_ids"]]

    # Save
    out_df.to_parquet(out_path, index=False)

    return out_df
