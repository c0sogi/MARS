import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from library.config import Config
from library.utils import get_pos_tagger


class Vocabulary:
    """
    Manages the vocabulary for the model, handling the mapping between words and indices.
    Supports special tokens and saving/loading from disk.
    """

    def __init__(self):
        self.stoi = {}  # String to Integer
        self.itos = {}  # Integer to String

        # Initialize with special tokens from Config
        self.specials = Config.SPECIAL_TOKENS
        for i, token in enumerate(self.specials):
            self.stoi[token] = i
            self.itos[i] = token

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        """
        Returns the index of the token. Returns UNK_IDX if token is not found.
        """
        return self.stoi.get(token, Config.UNK_IDX)

    def lookup_token(self, idx):
        """
        Returns the token corresponding to the index.
        """
        return self.itos.get(idx, Config.UNK_TOKEN)

    def build(self, sentences):
        """
        Builds the vocabulary from a list of sentences.

        Args:
            sentences (list of str): List of sentences to build vocabulary from.
        """
        print("Building vocabulary...")
        counter = Counter()
        for sentence in sentences:
            if not isinstance(sentence, str):
                continue
            # Simple whitespace tokenization as per dataset analysis
            tokens = sentence.split()
            counter.update(tokens)

        # Filter by minimum frequency
        # We reserve space for special tokens, so we take VOCAB_SIZE - len(specials)
        max_vocab_words = Config.VOCAB_SIZE - len(self.specials)

        # Get most common words
        common_words = counter.most_common(max_vocab_words)

        # Add to vocabulary
        idx = len(self.specials)
        for word, count in common_words:
            if count >= Config.MIN_FREQ:
                if word not in self.stoi:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        print(f"Vocabulary built. Size: {len(self)}")

    def save(self, path):
        """
        Saves the vocabulary (itos list) to a numpy file.
        """
        # We save the list of words ordered by index.
        # Indices are 0 to len(self)-1.
        vocab_list = [self.itos[i] for i in range(len(self))]
        np.save(path, np.array(vocab_list))
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """
        Loads the vocabulary from a numpy file.
        """
        vocab_list = np.load(path, allow_pickle=True)
        self.stoi = {}
        self.itos = {}

        for i, word in enumerate(vocab_list):
            # Ensure word is string
            word = str(word)
            self.stoi[word] = i
            self.itos[i] = word
        print(f"Vocabulary loaded from {path}. Size: {len(self)}")


def _build_pos_map_internal(vocab, sentences, sample_size=200000):
    """
    Internal function to build the word_id -> pos_tag_id mapping.

    Args:
        vocab (Vocabulary): The built vocabulary.
        sentences (list): List of training sentences.
        sample_size (int): Number of sentences to use for POS tagging.

    Returns:
        tuple: (pos_map_array, pos_tags_list)
    """
    print(f"Building POS Map using a sample of {sample_size} sentences...")

    # Initialize tagger
    tagger = get_pos_tagger()

    # Sample sentences to save time
    if len(sentences) > sample_size:
        # Deterministic sampling handled by caller or just slice here since order is fixed in list
        sample_sentences = sentences[:sample_size]
    else:
        sample_sentences = sentences

    # Store counts of tags for each word in vocab
    # word_id -> {tag_name: count}
    word_tag_counts = defaultdict(Counter)

    # Process sentences
    for sent in sample_sentences:
        if not isinstance(sent, str):
            continue
        tokens = sent.split()
        tags = tagger(tokens)  # Returns list of (word, tag)

        for word, tag in tags:
            if word in vocab.stoi:
                wid = vocab.stoi[word]
                word_tag_counts[wid][tag] += 1

    # Define POS Tag Set
    # We include a PAD tag at index 0.
    # Universal tagset: ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRT, PRON, VERB, ., X
    # We collect all observed tags to be safe, plus ensure standard ones exist.
    all_observed_tags = set()
    for counts in word_tag_counts.values():
        all_observed_tags.update(counts.keys())

    # Sort tags for determinism
    sorted_tags = sorted(list(all_observed_tags))

    # Create POS vocab
    # 0 is reserved for PAD (Config.PAD_TOKEN equivalent for tags)
    pos_stoi = {Config.PAD_TOKEN: 0}
    pos_itos = {0: Config.PAD_TOKEN}

    current_idx = 1
    for tag in sorted_tags:
        pos_stoi[tag] = current_idx
        pos_itos[current_idx] = tag
        current_idx += 1

    # Ensure we don't exceed Config.NUM_POS_TAGS
    if len(pos_stoi) > Config.NUM_POS_TAGS:
        print(
            f"Warning: Number of POS tags ({len(pos_stoi)}) exceeds Config.NUM_POS_TAGS ({Config.NUM_POS_TAGS}). Truncating."
        )
        # This is rare with Universal Tagset (12 tags), but safety check.

    # Create the mapping array: index = word_id, value = pos_tag_id
    vocab_size = len(vocab)
    pos_map = np.zeros(vocab_size, dtype=np.int64)

    # Fill mapping
    for wid in range(vocab_size):
        if wid in word_tag_counts:
            # Get most frequent tag
            most_common_tag = word_tag_counts[wid].most_common(1)[0][0]
            if most_common_tag in pos_stoi:
                pos_map[wid] = pos_stoi[most_common_tag]
            else:
                pos_map[wid] = 0  # Fallback to PAD/Unknown
        else:
            # Word not seen in POS sample (likely rare or special token)
            # Assign 'X' (Other) if available, else 0
            if "X" in pos_stoi:
                pos_map[wid] = pos_stoi["X"]
            else:
                pos_map[wid] = 0

    # Handle Special Tokens explicitly if needed
    # GAP, SOS, EOS, PAD, UNK usually don't have standard POS tags.
    # We leave them as 0 (PAD) or mapped to 'X' via the logic above if they appeared in sample.

    print(f"POS Map built. Found {len(pos_stoi)} unique tags.")

    # Return the map and the list of tags (for reconstruction)
    pos_tags_list = [pos_itos[i] for i in range(len(pos_itos))]

    return pos_map, pos_tags_list


def load_or_build_artifacts(load_cached_data=True):
    """
    Orchestrates the loading or creation of the Vocabulary and POS Mapping.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (vocab_instance, pos_map_numpy_array, pos_tags_list)
    """
    vocab_path = Config.VOCAB_SAVE_PATH
    pos_map_path = Config.POS_MAP_SAVE_PATH
    pos_tags_path = os.path.join(Config.WORKING_DIR, "pos_tags.npy")

    vocab = Vocabulary()

    # Check if files exist
    files_exist = (
        os.path.exists(vocab_path)
        and os.path.exists(pos_map_path)
        and os.path.exists(pos_tags_path)
    )

    if load_cached_data and files_exist:
        print("Loading artifacts from cache...")
        vocab.load(vocab_path)
        pos_map = np.load(pos_map_path)
        pos_tags = np.load(pos_tags_path, allow_pickle=True).tolist()
        return vocab, pos_map, pos_tags

    # If not loaded, build from scratch
    print("Building artifacts from scratch...")

    # Load training data
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Ensure strings
    sentences = df["sentence"].dropna().astype(str).tolist()

    # 1. Build Vocabulary
    vocab.build(sentences)
    vocab.save(vocab_path)

    # 2. Build POS Map
    # Use a subset for POS tagging to fit in time limits
    pos_map, pos_tags = _build_pos_map_internal(vocab, sentences, sample_size=200000)

    # Save POS artifacts
    np.save(pos_map_path, pos_map)
    np.save(pos_tags_path, np.array(pos_tags))
    print(f"POS Map saved to {pos_map_path}")

    return vocab, pos_map, pos_tags
