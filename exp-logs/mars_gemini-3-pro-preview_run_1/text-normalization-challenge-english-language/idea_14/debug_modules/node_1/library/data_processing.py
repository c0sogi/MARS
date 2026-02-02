import os
import json
import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
from collections import Counter, defaultdict
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_regex_features, set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class Vocab:
    """
    Manages vocabularies for Words, Characters, and Classes.
    """

    def __init__(self):
        self.word2id = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
        self.id2word = {0: Config.PAD_TOKEN, 1: Config.UNK_TOKEN}

        self.char2id = {
            Config.PAD_TOKEN: 0,
            Config.UNK_TOKEN: 1,
            Config.SOS_TOKEN: 2,
            Config.EOS_TOKEN: 3,
        }
        self.id2char = {
            0: Config.PAD_TOKEN,
            1: Config.UNK_TOKEN,
            2: Config.SOS_TOKEN,
            3: Config.EOS_TOKEN,
        }

        self.class2id = {
            Config.PAD_TOKEN: 0
        }  # 0 is padding for class labels if needed, though usually we use Ignore Index
        self.id2class = {0: Config.PAD_TOKEN}

        self.word_vocab_built = False
        self.char_vocab_built = False
        self.class_vocab_built = False

    def build_from_data(self, df):
        # 1. Class Vocab
        unique_classes = sorted(df["class"].unique().tolist())
        # Start from 1, 0 is PAD
        for i, cls in enumerate(unique_classes, start=1):
            self.class2id[cls] = i
            self.id2class[i] = cls
        self.class_vocab_built = True

        # 2. Word Vocab
        word_counts = Counter(df["before"].astype(str).tolist())
        most_common = word_counts.most_common(Config.MAX_WORD_VOCAB_SIZE)
        for i, (word, count) in enumerate(most_common, start=len(self.word2id)):
            if count >= Config.MIN_WORD_FREQ:
                self.word2id[word] = i
                self.id2word[i] = word
        self.word_vocab_built = True

        # 3. Char Vocab (from both before and after)
        # We need chars for input (before) and target (after) for Seq2Seq
        all_text = "".join(
            df["before"].astype(str).tolist() + df["after"].astype(str).tolist()
        )
        unique_chars = sorted(list(set(all_text)))
        for i, char in enumerate(unique_chars, start=len(self.char2id)):
            self.char2id[char] = i
            self.id2char[i] = char
        self.char_vocab_built = True

    def save(self):
        with open(Config.VOCAB_WORDS_PATH, "w") as f:
            json.dump(self.word2id, f)
        with open(Config.VOCAB_CHARS_PATH, "w") as f:
            json.dump(self.char2id, f)
        with open(Config.VOCAB_CLASSES_PATH, "w") as f:
            json.dump(self.class2id, f)

    def load(self):
        if os.path.exists(Config.VOCAB_WORDS_PATH):
            with open(Config.VOCAB_WORDS_PATH, "r") as f:
                self.word2id = json.load(f)
            self.id2word = {int(k): v for v, k in self.word2id.items()}
            self.word_vocab_built = True

        if os.path.exists(Config.VOCAB_CHARS_PATH):
            with open(Config.VOCAB_CHARS_PATH, "r") as f:
                self.char2id = json.load(f)
            self.id2char = {int(k): v for v, k in self.char2id.items()}
            self.char_vocab_built = True

        if os.path.exists(Config.VOCAB_CLASSES_PATH):
            with open(Config.VOCAB_CLASSES_PATH, "r") as f:
                self.class2id = json.load(f)
            self.id2class = {int(k): v for v, k in self.class2id.items()}
            self.class_vocab_built = True

    def __len__(self):
        return len(self.word2id)


class BPETokenizer:
    def __init__(self):
        self.model_prefix = Config.BPE_MODEL_PREFIX
        self.model_path = self.model_prefix + ".model"
        self.sp = spm.SentencePieceProcessor()

    def train(self, text_iterator, vocab_size=None):
        if vocab_size is None:
            vocab_size = Config.BPE_VOCAB_SIZE

        # SentencePiece needs a file
        temp_file = os.path.join(Config.WORKING_DIR, "bpe_corpus.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in text_iterator:
                f.write(str(text) + "\n")

        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=self.model_prefix,
            vocab_size=vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=Config.PAD_TOKEN,
            unk_piece=Config.UNK_TOKEN,
            bos_piece=Config.SOS_TOKEN,
            eos_piece=Config.EOS_TOKEN,
            model_type="bpe",
        )
        self.sp.load(self.model_path)
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def load(self):
        if os.path.exists(self.model_path):
            self.sp.load(self.model_path)
        else:
            raise FileNotFoundError(f"BPE model not found at {self.model_path}")

    def encode(self, text):
        return self.sp.encode_as_ids(str(text))

    def __len__(self):
        return self.sp.get_piece_size()


class KnowledgeBase:
    """
    Stores deterministic mappings and computes class priors.
    """

    def __init__(self, vocab_classes):
        self.vocab_classes = vocab_classes
        self.lookup = {}  # (token, class_name) -> normalized_text
        self.priors = {}  # token -> np.array(num_classes)
        self.num_classes = len(vocab_classes)

    def build(self, df):
        # 1. Build Lookup
        # We want to store (token, class) -> after
        # Since the same token+class might theoretically map to different things (unlikely but possible),
        # we take the most frequent one or just the last one.
        # Group by token, class, after and count
        print("Building Knowledge Base Lookup...")
        counts = (
            df.groupby(["before", "class", "after"]).size().reset_index(name="count")
        )
        # Sort by count desc to keep most frequent
        counts = counts.sort_values("count", ascending=False)

        for _, row in counts.iterrows():
            key = (str(row["before"]), str(row["class"]))
            if key not in self.lookup:
                self.lookup[key] = str(row["after"])

        # 2. Build Priors
        print("Building Knowledge Base Priors...")
        # Count class occurrences per token
        token_class_counts = (
            df.groupby(["before", "class"]).size().reset_index(name="count")
        )

        # Aggregate to dict
        temp_priors = defaultdict(lambda: defaultdict(int))
        for _, row in token_class_counts.iterrows():
            temp_priors[str(row["before"])][str(row["class"])] = row["count"]

        # Convert to probability vectors
        # Vector size is len(vocab_classes) (including PAD at 0)
        # We will index by class_id
        for token, cls_counts in temp_priors.items():
            total = sum(cls_counts.values())
            vec = np.zeros(self.num_classes, dtype=np.float32)
            for cls_name, count in cls_counts.items():
                if cls_name in self.vocab_classes.class2id:
                    cid = self.vocab_classes.class2id[cls_name]
                    vec[cid] = count / total
            self.priors[token] = vec

    def save(self):
        # Save lookup as parquet
        lookup_data = []
        for (token, cls), after in self.lookup.items():
            lookup_data.append({"before": token, "class": cls, "after": after})
        df_lookup = pd.DataFrame(lookup_data)
        df_lookup.to_parquet(Config.KNOWLEDGE_BASE_PATH, index=False)

        # Save priors as parquet (token, vector_json or similar? No, parquet supports arrays usually,
        # but to be safe and simple, we can save token, class, prob and reconstruct)
        # Actually, saving the dense matrix is heavy. Let's save the sparse structure.
        # We'll save a dataframe: before, class, probability
        priors_data = []
        for token, vec in self.priors.items():
            # Extract non-zero for storage efficiency
            for cid, prob in enumerate(vec):
                if prob > 0:
                    priors_data.append({"before": token, "class_id": cid, "prob": prob})
        df_priors = pd.DataFrame(priors_data)
        df_priors.to_parquet(Config.PRIORS_PATH, index=False)

    def load(self):
        if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
            df_lookup = pd.read_parquet(Config.KNOWLEDGE_BASE_PATH)
            self.lookup = {}
            for _, row in df_lookup.iterrows():
                self.lookup[(row["before"], row["class"])] = row["after"]

        if os.path.exists(Config.PRIORS_PATH):
            df_priors = pd.read_parquet(Config.PRIORS_PATH)
            self.priors = {}
            # Group by token to reconstruct vectors efficiently
            # This might be slow in python loop.
            # Optimization: Use numpy grouping if possible, but dict is fine for loading.
            grouped = df_priors.groupby("before")
            for token, group in grouped:
                vec = np.zeros(self.num_classes, dtype=np.float32)
                for _, row in group.iterrows():
                    vec[int(row["class_id"])] = row["prob"]
                self.priors[token] = vec

    def get_prior(self, token):
        return self.priors.get(str(token), np.zeros(self.num_classes, dtype=np.float32))

    def get_normalization(self, token, cls_name):
        return self.lookup.get((str(token), str(cls_name)), None)


class TaggerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data[idx] is a dictionary of tensors/arrays
        item = self.data[idx]
        return {
            "word_ids": torch.tensor(item["word_ids"], dtype=torch.long),
            "bpe_ids": torch.tensor(
                item["bpe_ids"], dtype=torch.long
            ),  # Already padded or fixed length?
            # BPE is variable length per token. We usually pool them.
            # To handle this in a batch, we can pad BPEs to a fixed small number (e.g. 5) per token
            # OR we can just use the first few.
            # For simplicity in this implementation, let's assume we pre-processed BPEs
            # to be a fixed length per token or we handle it here.
            # Let's check how we process it.
            "char_ids": torch.tensor(item["char_ids"], dtype=torch.long),
            "regex_features": torch.tensor(item["regex_features"], dtype=torch.float32),
            "prior_features": torch.tensor(item["prior_features"], dtype=torch.float32),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "mask": torch.tensor(item["mask"], dtype=torch.bool),
        }


class Seq2SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src_char_ids": torch.tensor(item["src_char_ids"], dtype=torch.long),
            "tgt_char_ids": torch.tensor(item["tgt_char_ids"], dtype=torch.long),
            "class_id": torch.tensor(item["class_id"], dtype=torch.long),
        }


def pad_sequence_fixed(seq, max_len, pad_val=0):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_val] * (max_len - len(seq))


def process_tagger_data(df, vocab, bpe, kb, max_sent_len=Config.MAX_SENT_LEN):
    """
    Groups data by sentence and generates features.
    """
    # Sort by sentence_id and token_id to ensure order
    df = df.sort_values(["sentence_id", "token_id"])

    # Group by sentence
    grouped = df.groupby("sentence_id")

    processed_data = []

    # Pre-fetch vocab lookups for speed
    word2id = vocab.word2id
    unk_word = vocab.word2id[Config.UNK_TOKEN]
    char2id = vocab.char2id
    unk_char = vocab.char2id[Config.UNK_TOKEN]
    cls2id = vocab.class2id

    # Iterate over sentences
    for _, group in grouped:
        tokens = group["before"].astype(str).tolist()
        classes = (
            group["class"].tolist()
            if "class" in group.columns
            else [Config.PAD_TOKEN] * len(tokens)
        )

        # Truncate if too long
        if len(tokens) > max_sent_len:
            tokens = tokens[:max_sent_len]
            classes = classes[:max_sent_len]

        seq_len = len(tokens)

        # Initialize arrays
        word_ids = []
        bpe_ids_list = []  # List of lists
        char_ids_list = []  # List of lists
        regex_feats = []
        prior_feats = []
        labels = []

        for i in range(seq_len):
            token = tokens[i]
            cls = classes[i]

            # Word ID
            word_ids.append(word2id.get(token, unk_word))

            # BPE IDs (Mean pooling happens in model, here we store indices)
            # We fix BPE length per token to, say, 8 subwords for tensor stacking
            b_ids = bpe.encode(token)
            bpe_ids_list.append(
                pad_sequence_fixed(b_ids, 8, 0)
            )  # 0 is usually unk or pad in SP? SP uses 0 for unk usually, but we need pad.
            # Actually SP vocab: <unk>=0, <s>=1, </s>=2. We should check.
            # We'll assume 0 is safe or use a specific pad id.
            # In our Vocab class we defined PAD=0. In SP, we need to be careful.
            # Let's just use 0 and ensure the embedding layer handles it.

            # Char IDs (for CharCNN)
            # Fix char length per token
            c_ids = [char2id.get(c, unk_char) for c in token]
            char_ids_list.append(pad_sequence_fixed(c_ids, Config.MAX_TOKEN_LEN, 0))

            # Regex
            regex_feats.append(get_regex_features(token))

            # Prior
            prior_feats.append(kb.get_prior(token))

            # Label
            labels.append(cls2id.get(cls, 0))

        # Padding to MAX_SENT_LEN
        pad_len = max_sent_len - seq_len

        # Pad Word IDs
        word_ids = word_ids + [0] * pad_len

        # Pad BPE (List of lists -> Tensor)
        # Pad with [0,0,0...]
        for _ in range(pad_len):
            bpe_ids_list.append([0] * 8)

        # Pad Char (List of lists)
        for _ in range(pad_len):
            char_ids_list.append([0] * Config.MAX_TOKEN_LEN)

        # Pad Regex (Vector of 0s)
        for _ in range(pad_len):
            regex_feats.append(np.zeros(Config.NUM_REGEX_FEATURES, dtype=np.float32))

        # Pad Prior (Vector of 0s)
        for _ in range(pad_len):
            prior_feats.append(np.zeros(kb.num_classes, dtype=np.float32))

        # Pad Labels
        labels = labels + [0] * pad_len

        # Mask
        mask = [True] * seq_len + [False] * pad_len

        processed_data.append(
            {
                "word_ids": np.array(word_ids, dtype=np.int64),
                "bpe_ids": np.array(bpe_ids_list, dtype=np.int64),
                "char_ids": np.array(char_ids_list, dtype=np.int64),
                "regex_features": np.array(regex_feats, dtype=np.float32),
                "prior_features": np.array(prior_feats, dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64),
                "mask": np.array(mask, dtype=np.bool_),
            }
        )

    return processed_data


def process_seq2seq_data(df, vocab):
    """
    Filters for changed tokens and generates char-level pairs.
    """
    # Filter changed
    changed = df[df["before"] != df["after"]].copy()

    data = []
    char2id = vocab.char2id
    unk_char = vocab.char2id[Config.UNK_TOKEN]
    cls2id = vocab.class2id

    for _, row in changed.iterrows():
        src = str(row["before"])
        tgt = str(row["after"])
        cls = str(row["class"])

        # Source Chars
        src_ids = [char2id.get(c, unk_char) for c in src]
        src_ids = pad_sequence_fixed(src_ids, Config.MAX_TOKEN_LEN, 0)

        # Target Chars (Add SOS/EOS)
        tgt_ids = (
            [vocab.char2id[Config.SOS_TOKEN]]
            + [char2id.get(c, unk_char) for c in tgt]
            + [vocab.char2id[Config.EOS_TOKEN]]
        )
        tgt_ids = pad_sequence_fixed(tgt_ids, Config.SEQ2SEQ_MAX_OUTPUT_LEN, 0)

        data.append(
            {
                "src_char_ids": np.array(src_ids, dtype=np.int64),
                "tgt_char_ids": np.array(tgt_ids, dtype=np.int64),
                "class_id": cls2id.get(cls, 0),
            }
        )

    return data


def load_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Main function to load and process all data.
    """
    print(f"Loading data (Debug={debug})...")

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV, keep_default_na=False, dtype=str)
    df_val = pd.read_csv(Config.VAL_CSV, keep_default_na=False, dtype=str)
    df_test = pd.read_csv(Config.TEST_CSV, keep_default_na=False, dtype=str)

    if debug:
        # Filter by sentence_id to keep integrity
        train_sents = df_train["sentence_id"].unique()[: Config.DEBUG_SIZE]
        df_train = df_train[df_train["sentence_id"].isin(train_sents)]
        val_sents = df_val["sentence_id"].unique()[: Config.DEBUG_SIZE // 5]
        df_val = df_val[df_val["sentence_id"].isin(val_sents)]
        test_sents = df_test["sentence_id"].unique()[: Config.DEBUG_SIZE // 5]
        df_test = df_test[df_test["sentence_id"].isin(test_sents)]
        print(
            f"Debug Mode: Train {len(df_train)}, Val {len(df_val)}, Test {len(df_test)}"
        )

    # 2. Init Vocab & Tokenizer
    vocab = Vocab()
    bpe = BPETokenizer()

    if (
        load_cached_data
        and os.path.exists(Config.VOCAB_WORDS_PATH)
        and os.path.exists(bpe.model_path)
    ):
        print("Loading cached vocab and BPE model...")
        vocab.load()
        bpe.load()
    else:
        print("Building vocab and training BPE...")
        vocab.build_from_data(df_train)
        vocab.save()
        bpe.train(df_train["before"].astype(str))

    # 3. Init Knowledge Base
    kb = KnowledgeBase(vocab)
    if (
        load_cached_data
        and os.path.exists(Config.KNOWLEDGE_BASE_PATH)
        and os.path.exists(Config.PRIORS_PATH)
    ):
        print("Loading cached Knowledge Base...")
        kb.load()
    else:
        print("Building Knowledge Base...")
        kb.build(df_train)
        kb.save()

    # 4. Process Tagger Data
    tagger_train_path = Config.TAGGER_TRAIN_DATA
    tagger_val_path = Config.TAGGER_VAL_DATA

    if (
        load_cached_data
        and os.path.exists(tagger_train_path)
        and os.path.exists(tagger_val_path)
    ):
        print("Loading cached Tagger data...")
        tagger_train_data = torch.load(tagger_train_path)
        tagger_val_data = torch.load(tagger_val_path)
    else:
        print("Processing Tagger train data...")
        tagger_train_data = process_tagger_data(df_train, vocab, bpe, kb)
        torch.save(tagger_train_data, tagger_train_path)

        print("Processing Tagger val data...")
        tagger_val_data = process_tagger_data(df_val, vocab, bpe, kb)
        torch.save(tagger_val_data, tagger_val_path)

    # 5. Process Seq2Seq Data
    seq2seq_train_path = Config.SEQ2SEQ_TRAIN_DATA
    seq2seq_val_path = Config.SEQ2SEQ_VAL_DATA

    if (
        load_cached_data
        and os.path.exists(seq2seq_train_path)
        and os.path.exists(seq2seq_val_path)
    ):
        print("Loading cached Seq2Seq data...")
        seq2seq_train_data = torch.load(seq2seq_train_path)
        seq2seq_val_data = torch.load(seq2seq_val_path)
    else:
        print("Processing Seq2Seq train data...")
        seq2seq_train_data = process_seq2seq_data(df_train, vocab)
        torch.save(seq2seq_train_data, seq2seq_train_path)

        print("Processing Seq2Seq val data...")
        seq2seq_val_data = process_seq2seq_data(df_val, vocab)
        torch.save(seq2seq_val_data, seq2seq_val_path)

    # 6. Process Test Data (Only Tagger features needed)
    # We don't cache test data usually as it's for inference, but we can return the raw df
    # or process it. The inference pipeline usually processes it batch by batch.
    # Here we return the raw test df and let the inference script handle it using the vocab/kb.

    return {
        "vocab": vocab,
        "bpe": bpe,
        "kb": kb,
        "tagger_train": TaggerDataset(tagger_train_data),
        "tagger_val": TaggerDataset(tagger_val_data),
        "seq2seq_train": Seq2SeqDataset(seq2seq_train_data),
        "seq2seq_val": Seq2SeqDataset(seq2seq_val_data),
        "test_df": df_test,
    }
