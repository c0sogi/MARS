import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import (
    save_parquet,
    load_parquet,
    setup_logger,
    ensure_dir,
    save_npy,
    load_npy,
)
from library.features import RegexFeatureExtractor, GlobalPriorManager
from library.vocab import VocabManager

logger = setup_logger("data_loader")


class KnowledgeBase:
    """
    Deterministic memory mapping (token, class) -> normalization.
    """

    def __init__(self):
        self.kb = {}
        self.path = Config.KNOWLEDGE_BASE_PATH

    def build(self, df, save=True):
        """
        Builds the knowledge base from a dataframe.
        Expects columns: 'before', 'class', 'after'.
        """
        logger.info("Building Knowledge Base...")
        # Efficient approach:
        # 1. Count occurrences of (before, class, after)
        counts = (
            df.groupby(["before", "class", "after"]).size().reset_index(name="count")
        )
        # 2. Sort by count descending
        counts = counts.sort_values("count", ascending=False)
        # 3. Drop duplicates on (before, class), keeping the first (highest count)
        kb_df = counts.drop_duplicates(subset=["before", "class"])

        # Convert to dict
        self.kb = {}
        for row in kb_df.itertuples():
            self.kb[(str(row.before), str(row._2))] = str(row.after)

        logger.info(f"Knowledge Base built with {len(self.kb)} entries.")

        if save:
            self.save()

    def lookup(self, token, token_class):
        """
        Returns the normalized text or None if not found.
        """
        return self.kb.get((str(token), str(token_class)))

    def save(self):
        # Save as parquet for efficiency
        data = []
        for (token, cls), after in self.kb.items():
            data.append({"before": token, "class": cls, "after": after})
        df = pd.DataFrame(data)
        save_parquet(df, self.path)
        logger.info(f"Knowledge Base saved to {self.path}")

    def load(self):
        if os.path.exists(self.path):
            df = load_parquet(self.path)
            self.kb = {}
            # Ensure strings
            df["before"] = df["before"].astype(str)
            df["class"] = df["class"].astype(str)
            df["after"] = df["after"].astype(str)

            # Reconstruct dictionary
            # Using zip is faster than itertuples for simple construction
            self.kb = dict(zip(zip(df["before"], df["class"]), df["after"]))

            logger.info(f"Knowledge Base loaded with {len(self.kb)} entries.")
            return True
        return False


class TaggerDataset(Dataset):
    """
    Dataset for the Penta-Hybrid Tagger.
    Groups tokens by sentence.
    """

    def __init__(
        self,
        metadata_path,
        vocab_manager,
        prior_manager,
        mode="train",
        load_cached_data=True,
    ):
        self.vocab_manager = vocab_manager
        self.prior_manager = prior_manager
        self.mode = mode
        self.regex_extractor = RegexFeatureExtractor()

        self.word_vocab = vocab_manager.get_word_vocab()
        self.bpe_tokenizer = vocab_manager.get_bpe_tokenizer()
        self.char_vocab = vocab_manager.get_char_vocab()
        self.class_vocab = vocab_manager.get_class_vocab()

        # Load and Group Data
        self.data = self._load_data(metadata_path, load_cached_data)

    def _load_data(self, path, load_cached_data):
        # Determine cache path based on input filename
        base_name = os.path.basename(path).replace(".csv", "")
        cache_path = os.path.join(Config.WORK_DIR, f"{base_name}_grouped.parquet")

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading grouped data from {cache_path}...")
            return load_parquet(cache_path)

        logger.info(f"Processing raw data from {path}...")
        df = pd.read_csv(path)

        # Ensure strings
        df["before"] = df["before"].astype(str)
        if "class" in df.columns:
            df["class"] = df["class"].astype(str)
        if "after" in df.columns:
            df["after"] = df["after"].astype(str)

        # Group by sentence_id
        # We need to preserve order of token_id
        df = df.sort_values(["sentence_id", "token_id"])

        agg_dict = {"before": list, "id": list}
        if "class" in df.columns:
            agg_dict["class"] = list
        if "after" in df.columns:
            agg_dict["after"] = list

        grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

        # Save cache
        save_parquet(grouped, cache_path)
        return grouped

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row["before"]

        # Truncate to MAX_SEQ_LEN
        if len(tokens) > Config.MAX_SEQ_LEN:
            tokens = tokens[: Config.MAX_SEQ_LEN]

        # 1. Word IDs
        word_ids = [self.word_vocab[t] for t in tokens]

        # 2. BPE IDs (List of Lists)
        bpe_ids = [self.bpe_tokenizer.encode(t) for t in tokens]

        # 3. Char IDs (List of Lists)
        char_ids = []
        for t in tokens:
            # Truncate chars per token
            chars = list(t)[: Config.MAX_CHAR_LEN]
            c_ids = [self.char_vocab[c] for c in chars]
            char_ids.append(c_ids)

        # 4. Regex Features
        regex_feats = self.regex_extractor.extract_batch(tokens)

        # 5. Global Prior Features
        prior_feats = np.array([self.prior_manager.get_prior(t) for t in tokens])

        result = {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "bpe_ids": bpe_ids,  # List of lists, handled in collate
            "char_ids": char_ids,  # List of lists, handled in collate
            "regex_feats": torch.tensor(regex_feats, dtype=torch.float32),
            "prior_feats": torch.tensor(prior_feats, dtype=torch.float32),
            "ids": row["id"][: len(tokens)],
        }

        if self.mode != "test":
            classes = row["class"][: len(tokens)]
            class_ids = [self.class_vocab[c] for c in classes]
            result["targets"] = torch.tensor(class_ids, dtype=torch.long)

        return result

    @staticmethod
    def collate_fn(batch):
        # batch is list of dicts

        # 1. Pad Word IDs (Batch, Seq)
        word_ids = [item["word_ids"] for item in batch]
        word_ids_padded = pad_sequence(
            word_ids, batch_first=True, padding_value=0
        )  # 0 is <pad>

        # 2. Pad Targets (Batch, Seq)
        if "targets" in batch[0]:
            targets = [item["targets"] for item in batch]
            targets_padded = pad_sequence(
                targets, batch_first=True, padding_value=-1
            )  # -1 for ignore index
        else:
            targets_padded = None

        # 3. Pad Regex & Priors (Batch, Seq, Dim)
        regex_feats = [item["regex_feats"] for item in batch]
        regex_padded = pad_sequence(regex_feats, batch_first=True, padding_value=0.0)

        prior_feats = [item["prior_feats"] for item in batch]
        prior_padded = pad_sequence(prior_feats, batch_first=True, padding_value=0.0)

        # 4. Pad BPE & Char (Batch, Seq, Sub_Len)
        # Find max dims
        max_seq = word_ids_padded.size(1)
        max_bpe = 0
        max_char = 0

        for item in batch:
            for bpe in item["bpe_ids"]:
                max_bpe = max(max_bpe, len(bpe))
            for chars in item["char_ids"]:
                max_char = max(max_char, len(chars))

        # Ensure at least 1 to avoid zero-size tensor errors
        max_bpe = max(max_bpe, 1)
        max_char = max(max_char, 1)

        batch_size = len(batch)

        bpe_padded = torch.zeros((batch_size, max_seq, max_bpe), dtype=torch.long)
        char_padded = torch.zeros((batch_size, max_seq, max_char), dtype=torch.long)

        # Fill
        for i, item in enumerate(batch):
            seq_len = len(item["word_ids"])
            for j in range(seq_len):
                # BPE
                b_ids = item["bpe_ids"][j]
                if len(b_ids) > 0:
                    bpe_padded[i, j, : len(b_ids)] = torch.tensor(
                        b_ids, dtype=torch.long
                    )

                # Char
                c_ids = item["char_ids"][j]
                if len(c_ids) > 0:
                    char_padded[i, j, : len(c_ids)] = torch.tensor(
                        c_ids, dtype=torch.long
                    )

        # Mask
        mask = (word_ids_padded != 0).bool()

        return {
            "word_ids": word_ids_padded,
            "bpe_ids": bpe_padded,
            "char_ids": char_padded,
            "regex_feats": regex_padded,
            "prior_feats": prior_padded,
            "targets": targets_padded,
            "mask": mask,
            "ids": [item["ids"] for item in batch],  # List of lists of strings
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Transformer Fallback (Seq2Seq).
    Filters for 'changed' tokens only (before != after).
    """

    def __init__(
        self, metadata_path, vocab_manager, mode="train", load_cached_data=True
    ):
        self.vocab_manager = vocab_manager
        self.mode = mode
        self.char_vocab = vocab_manager.get_char_vocab()
        self.class_vocab = vocab_manager.get_class_vocab()

        self.data = self._load_data(metadata_path, load_cached_data)

    def _load_data(self, path, load_cached_data):
        base_name = os.path.basename(path).replace(".csv", "")
        cache_path = os.path.join(Config.WORK_DIR, f"{base_name}_seq2seq.parquet")

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading seq2seq data from {cache_path}...")
            return load_parquet(cache_path)

        logger.info(f"Processing seq2seq data from {path}...")
        df = pd.read_csv(path)

        # Ensure strings
        df["before"] = df["before"].astype(str)
        df["class"] = df["class"].astype(str)
        df["after"] = df["after"].astype(str)

        # Filter: keep only changed tokens
        df = df[df["before"] != df["after"]].copy()

        # Save cache
        save_parquet(df, cache_path)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = row["before"]
        tgt_text = row["after"]
        cls_name = row["class"]

        # Truncate sequences to fit PositionalEncoding limits
        if len(src_text) > Config.MAX_SEQ2SEQ_LEN:
            src_text = src_text[: Config.MAX_SEQ2SEQ_LEN]
        if len(tgt_text) > Config.MAX_SEQ2SEQ_LEN:
            tgt_text = tgt_text[: Config.MAX_SEQ2SEQ_LEN]

        # Encode Source (Chars)
        src_ids = [self.char_vocab[c] for c in src_text]

        # Encode Target (Chars)
        # <sos> and <eos> for decoder
        tgt_ids = (
            [self.char_vocab["<sos>"]]
            + [self.char_vocab[c] for c in tgt_text]
            + [self.char_vocab["<eos>"]]
        )

        # Class ID
        class_id = self.class_vocab[cls_name]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        src_ids = [item["src_ids"] for item in batch]
        tgt_ids = [item["tgt_ids"] for item in batch]
        class_ids = torch.stack([item["class_id"] for item in batch])

        # Pad
        src_padded = pad_sequence(src_ids, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=0)

        return {
            "src_ids": src_padded,
            "tgt_ids": tgt_padded,
            "class_ids": class_ids,
        }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to prepare all data and return dataloaders.
    """
    # 1. Prepare Vocabs
    vocab_manager = VocabManager()
    vocab_manager.build_or_load(load_cached_data=load_cached_data)

    # 2. Prepare Priors
    # Need to load train df to build priors if not cached
    prior_manager = GlobalPriorManager()
    if not os.path.exists(Config.PRIORS_PATH) or not load_cached_data:
        train_df = pd.read_csv(Config.TRAIN_FILE)
        prior_manager.build_or_load(train_df, load_cached_data=load_cached_data)
    else:
        # Load from cache without df
        prior_manager.build_or_load(None, load_cached_data=True)

    # 3. Build Knowledge Base
    kb = KnowledgeBase()
    if not kb.load() or not load_cached_data:
        train_df = pd.read_csv(Config.TRAIN_FILE)
        kb.build(train_df)

    # 4. Tagger Datasets
    logger.info("Creating Tagger Datasets...")
    train_tagger_ds = TaggerDataset(
        Config.TRAIN_FILE,
        vocab_manager,
        prior_manager,
        mode="train",
        load_cached_data=load_cached_data,
    )
    val_tagger_ds = TaggerDataset(
        Config.VAL_FILE,
        vocab_manager,
        prior_manager,
        mode="val",
        load_cached_data=load_cached_data,
    )

    # 5. Seq2Seq Datasets
    logger.info("Creating Seq2Seq Datasets...")
    train_seq2seq_ds = Seq2SeqDataset(
        Config.TRAIN_FILE,
        vocab_manager,
        mode="train",
        load_cached_data=load_cached_data,
    )
    val_seq2seq_ds = Seq2SeqDataset(
        Config.VAL_FILE, vocab_manager, mode="val", load_cached_data=load_cached_data
    )

    # 6. DataLoaders
    train_tagger_loader = DataLoader(
        train_tagger_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=TaggerDataset.collate_fn,
        pin_memory=True,
    )

    val_tagger_loader = DataLoader(
        val_tagger_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=TaggerDataset.collate_fn,
        pin_memory=True,
    )

    train_seq2seq_loader = DataLoader(
        train_seq2seq_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=Seq2SeqDataset.collate_fn,
        pin_memory=True,
    )

    val_seq2seq_loader = DataLoader(
        val_seq2seq_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=Seq2SeqDataset.collate_fn,
        pin_memory=True,
    )

    return {
        "tagger": {"train": train_tagger_loader, "val": val_tagger_loader},
        "seq2seq": {"train": train_seq2seq_loader, "val": val_seq2seq_loader},
        "vocab_manager": vocab_manager,
        "prior_manager": prior_manager,
        "kb": kb,
    }
