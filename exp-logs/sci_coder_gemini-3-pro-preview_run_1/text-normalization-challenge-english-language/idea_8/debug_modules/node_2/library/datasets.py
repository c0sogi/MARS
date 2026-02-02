import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class TaggerDataset(Dataset):
    def __init__(self, df, vocab_tokens, vocab_chars, vocab_classes, split="train"):
        """
        Dataset for the Bi-LSTM Tagger.
        Expects a DataFrame grouped by sentence_id, where 'before', 'class', etc. are lists.
        """
        self.df = df
        self.vocab_tokens = vocab_tokens
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.split = split
        self.max_char_len = Config.MAX_CHAR_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row["before"]  # List of strings

        # 1. Token Embeddings (Word Level)
        token_ids = self.vocab_tokens.numericalize(tokens)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        # 2. Character Features (Char Level for CNN)
        # Returns a list of 1D tensors (one per token), to be padded in collate_fn
        char_ids_list = []
        for token in tokens:
            # Truncate to max char len for CNN efficiency
            chars = list(token)[: self.max_char_len]
            c_ids = self.vocab_chars.numericalize(chars)
            char_ids_list.append(torch.tensor(c_ids, dtype=torch.long))

        # 3. Targets (Classes)
        label_ids_tensor = None
        # Check if 'class' column exists and is valid (it might be missing in test set)
        if "class" in row and isinstance(row["class"], list):
            classes = row["class"]
            label_ids = self.vocab_classes.numericalize(classes)
            label_ids_tensor = torch.tensor(label_ids, dtype=torch.long)

        return {
            "token_ids": token_ids_tensor,
            "char_ids": char_ids_list,
            "label_ids": label_ids_tensor,
            "raw_tokens": tokens,
            "id": row["id"] if "id" in row else [],
        }


class TaggerCollator:
    def __init__(self, vocab_tokens, vocab_chars):
        """
        Collator for TaggerDataset. Handles 2D padding for tokens and 3D padding for characters.
        """
        self.token_pad_idx = vocab_tokens.stoi[Config.PAD_TOKEN]
        self.char_pad_idx = vocab_chars.stoi[Config.PAD_TOKEN]
        # Standard PyTorch ignore_index for CrossEntropyLoss
        self.label_pad_idx = -100

    def __call__(self, batch):
        # Extract fields
        token_ids = [item["token_ids"] for item in batch]
        char_ids_nested = [item["char_ids"] for item in batch]
        raw_tokens = [item["raw_tokens"] for item in batch]
        ids = [item["id"] for item in batch]

        # 1. Pad Tokens: (Batch, Max_Seq_Len)
        padded_tokens = pad_sequence(
            token_ids, batch_first=True, padding_value=self.token_pad_idx
        )

        # 2. Pad Labels: (Batch, Max_Seq_Len)
        padded_labels = None
        if batch[0]["label_ids"] is not None:
            label_ids = [item["label_ids"] for item in batch]
            padded_labels = pad_sequence(
                label_ids, batch_first=True, padding_value=self.label_pad_idx
            )

        # 3. Pad Characters: (Batch, Max_Seq_Len, Max_Char_Len)
        batch_size = len(batch)
        max_seq_len = padded_tokens.size(1)
        max_char_len = Config.MAX_CHAR_LEN

        # Initialize tensor with PAD value
        padded_chars = torch.full(
            (batch_size, max_seq_len, max_char_len), self.char_pad_idx, dtype=torch.long
        )

        # Fill with actual character indices
        for i, sent_chars in enumerate(char_ids_nested):
            for j, token_chars in enumerate(sent_chars):
                if j >= max_seq_len:
                    break
                length = min(len(token_chars), max_char_len)
                if length > 0:
                    padded_chars[i, j, :length] = token_chars[:length]

        # Lengths for packing sequences in LSTM
        lengths = torch.tensor([len(t) for t in token_ids], dtype=torch.long)

        return {
            "token_ids": padded_tokens,
            "char_ids": padded_chars,
            "label_ids": padded_labels,
            "lengths": lengths,
            "raw_tokens": raw_tokens,
            "ids": ids,
        }


class Seq2SeqDataset(Dataset):
    def __init__(self, df, vocab_chars, vocab_classes):
        """
        Dataset for the Transformer Seq2Seq Fallback.
        Expects a filtered DataFrame (one row per token pair).
        """
        self.df = df
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.sos_idx = vocab_chars.stoi[Config.SOS_TOKEN]
        self.eos_idx = vocab_chars.stoi[Config.EOS_TOKEN]
        self.max_len = Config.MAX_SEQ_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = row["before"]
        tgt_text = row["after"]
        cls_text = row["class"]

        # 1. Source: Character Indices
        src_indices = self.vocab_chars.numericalize(list(src_text))
        # Truncate to max length
        src_indices = src_indices[: self.max_len]
        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        # 2. Target: SOS + Character Indices + EOS
        tgt_indices = self.vocab_chars.numericalize(list(tgt_text))
        # Truncate to leave room for SOS and EOS
        tgt_indices = tgt_indices[: self.max_len - 2]
        tgt_full = [self.sos_idx] + tgt_indices + [self.eos_idx]
        tgt_tensor = torch.tensor(tgt_full, dtype=torch.long)

        # 3. Class Conditioning
        # Use .get() to handle potential unseen classes safely (though unlikely in train)
        cls_id = self.vocab_classes.stoi.get(cls_text, 0)
        cls_tensor = torch.tensor(cls_id, dtype=torch.long)

        return {"src_ids": src_tensor, "tgt_ids": tgt_tensor, "class_id": cls_tensor}


class Seq2SeqCollator:
    def __init__(self, vocab_chars):
        """
        Collator for Seq2SeqDataset. Handles padding for source and target sequences.
        """
        self.pad_idx = vocab_chars.stoi[Config.PAD_TOKEN]

    def __call__(self, batch):
        src_ids = [item["src_ids"] for item in batch]
        tgt_ids = [item["tgt_ids"] for item in batch]
        class_ids = [item["class_id"] for item in batch]

        # Pad sequences to max length in batch
        src_padded = pad_sequence(src_ids, batch_first=True, padding_value=self.pad_idx)
        tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=self.pad_idx)

        # Stack class indices
        class_stack = torch.stack(class_ids)

        return {
            "src_ids": src_padded,
            "tgt_ids": tgt_padded,
            "class_ids": class_stack,
            "src_lengths": torch.tensor([len(x) for x in src_ids], dtype=torch.long),
        }
