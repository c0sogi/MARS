import os
import json
import tempfile
import sentencepiece as spm
from typing import List, Dict, Optional
from library.config import Config
from library.utils import load_raw_data


class CharTokenizer:
    """
    Character-level tokenizer for the Encoder.
    Handles mapping of characters to IDs and vice versa.
    Includes special tokens for padding, unknown chars, sequence boundaries, and separators.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"

    def __init__(self):
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self.specials = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.SEP_TOKEN,
        ]
        # Initialize with specials
        for i, token in enumerate(self.specials):
            self.char2id[token] = i
            self.id2char[i] = token

    @property
    def pad_token_id(self) -> int:
        return self.char2id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.char2id[self.UNK_TOKEN]

    @property
    def sos_token_id(self) -> int:
        return self.char2id[self.SOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.char2id[self.EOS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.char2id[self.SEP_TOKEN]

    def fit(self, texts: List[str]):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        start_idx = len(self.specials)
        for i, char in enumerate(sorted_chars):
            # Avoid overwriting specials if they somehow appear in text
            if char not in self.char2id:
                idx = start_idx + i
                self.char2id[char] = idx
                self.id2char[idx] = char

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Converts a string to a list of token IDs.
        """
        # If the text contains the SEP token explicitly (as a string "<SEP>"),
        # we need to handle it. However, usually the construction happens
        # outside. Here we assume input is a raw string of characters unless
        # specific logic is applied.
        # For this task, the context construction likely concatenates strings.
        # We will treat the input 'text' as a sequence of characters.
        # If <SEP> is inserted as a special marker during data processing,
        # it should be passed as a list of tokens or handled here.
        # Given the standard char tokenizer usage, we iterate chars.

        ids = []
        if add_special_tokens:
            ids.append(self.sos_token_id)

        # Naive character iteration.
        # NOTE: If text contains "<SEP>", this loop breaks it into '<', 'S', 'E', 'P', '>'.
        # To support the idea's context structure properly, the caller should
        # construct the ID sequence manually using sep_token_id, or we provide a
        # method to encode a list of segments.
        # Here we implement standard string encoding.
        for char in text:
            ids.append(self.char2id.get(char, self.unk_token_id))

        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a list of token IDs back to a string.
        """
        chars = []
        for idx in ids:
            if skip_special_tokens and idx < len(self.specials):
                continue
            chars.append(self.id2char.get(idx, self.UNK_TOKEN))
        return "".join(chars)

    def save(self, path: str):
        """
        Saves the vocabulary to a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """
        Loads the vocabulary from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)
        # Reconstruct id2char
        self.id2char = {int(v): k for k, v in self.char2id.items()}

    def __len__(self):
        return len(self.char2id)


class BPETokenizer:
    """
    Wrapper around SentencePiece for the Decoder (Target Text).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.sp = spm.SentencePieceProcessor()
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, texts: List[str], model_prefix: str, vocab_size: int):
        """
        Trains a SentencePiece model.
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

        # Write texts to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        ) as tmp:
            for text in texts:
                tmp.write(str(text) + "\n")
            tmp_filename = tmp.name

        try:
            # Train SentencePiece
            # We use unigram or bpe. BPE is specified in the task description implicitly or explicitly.
            # "BPE (Subword) tokenization"
            spm.SentencePieceTrainer.train(
                input=tmp_filename,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<s>",
                eos_piece="</s>",
                user_defined_symbols=[],
            )
            # Load the trained model immediately
            self.load(model_prefix + ".model")
        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    def load(self, model_path: str):
        self.sp.Load(model_path)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if add_special_tokens:
            return [self.sp.bos_id()] + self.sp.EncodeAsIds(text) + [self.sp.eos_id()]
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        # SentencePiece DecodeIds handles skipping special tokens usually,
        # but we can filter if needed.
        # sp.DecodeIds automatically ignores PAD/BOS/EOS if configured,
        # but explicit filtering is safer if we want strict control.
        if skip_special_tokens:
            # Filter out special IDs manually to be safe
            filtered_ids = [
                i
                for i in ids
                if i not in [self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()]
            ]
            return self.sp.DecodeIds(filtered_ids)
        return self.sp.DecodeIds(ids)

    @property
    def pad_token_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_token_id(self) -> int:
        return self.sp.unk_id()

    @property
    def sos_token_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_token_id(self) -> int:
        return self.sp.eos_id()

    def __len__(self):
        return self.sp.GetPieceSize()


def build_tokenizers(load_cached_data: bool = True) -> (CharTokenizer, BPETokenizer):
    """
    Factory function to create, load, or train tokenizers based on configuration and cache.

    Args:
        load_cached_data: If True, attempts to load existing tokenizers from disk.
                          If False or if loading fails, trains from scratch.

    Returns:
        Tuple of (CharTokenizer, BPETokenizer)
    """
    char_tokenizer = CharTokenizer()
    bpe_tokenizer = BPETokenizer()

    # Define paths
    char_vocab_path = Config.CHAR_VOCAB_PATH
    bpe_model_prefix = Config.BPE_MODEL_PREFIX
    bpe_model_path = bpe_model_prefix + ".model"

    # Check if artifacts exist
    char_exists = os.path.exists(char_vocab_path)
    bpe_exists = os.path.exists(bpe_model_path)

    if load_cached_data and char_exists and bpe_exists:
        print("Loading tokenizers from cache...")
        try:
            char_tokenizer.load(char_vocab_path)
            bpe_tokenizer.load(bpe_model_path)
            print(f"CharTokenizer vocab size: {len(char_tokenizer)}")
            print(f"BPETokenizer vocab size: {len(bpe_tokenizer)}")
            return char_tokenizer, bpe_tokenizer
        except Exception as e:
            print(f"Failed to load cached tokenizers: {e}. Retraining...")

    print("Training tokenizers from scratch...")

    # Load training data
    # We need 'before' text for CharTokenizer and 'after' text for BPETokenizer
    print("Loading training data for tokenization...")
    df_train = load_raw_data("train")

    # 1. Train CharTokenizer
    # We use the 'before' column. We should also consider that context words
    # come from the same distribution, so 'before' covers all needed chars.
    print("Fitting CharTokenizer...")
    # Convert to string to handle potential non-string types safety
    raw_texts = df_train["before"].astype(str).tolist()
    char_tokenizer.fit(raw_texts)
    char_tokenizer.save(char_vocab_path)
    print(f"CharTokenizer trained. Vocab size: {len(char_tokenizer)}")

    # 2. Train BPETokenizer
    # We use the 'after' column (target Russian text).
    print("Training BPETokenizer...")
    target_texts = df_train["after"].astype(str).tolist()
    bpe_tokenizer.train(
        texts=target_texts,
        model_prefix=bpe_model_prefix,
        vocab_size=Config.BPE_VOCAB_SIZE,
    )
    print(f"BPETokenizer trained. Vocab size: {len(bpe_tokenizer)}")

    return char_tokenizer, bpe_tokenizer
