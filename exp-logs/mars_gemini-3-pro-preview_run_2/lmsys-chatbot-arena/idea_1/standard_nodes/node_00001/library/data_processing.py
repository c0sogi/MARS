import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_processing")


class TextVectorizer:
    """
    Wrapper for SentenceTransformer to encode text into embeddings.
    """

    def __init__(self, model_name=Config.TRANSFORMER_MODEL, device=Config.DEVICE):
        self.device = device
        logger.info(f"Loading SentenceTransformer: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = Config.MAX_LENGTH

    def encode(self, texts, batch_size=256, show_progress_bar=False):
        """
        Encodes a list of texts into numpy embeddings.
        """
        # Ensure texts is a list
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Handle potential NaNs by replacing with empty string
        texts = [str(t) if pd.notna(t) else "" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for better cosine/dot product properties in interaction features
        )
        return embeddings


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Chatbot Preference Prediction.
    Constructs feature vectors from pre-computed embeddings on the fly.
    """

    def __init__(self, prompt_emb, res_a_emb, res_b_emb, targets=None):
        self.prompt_emb = prompt_emb
        self.res_a_emb = res_a_emb
        self.res_b_emb = res_b_emb
        self.targets = targets

    def __len__(self):
        return len(self.prompt_emb)

    def __getitem__(self, idx):
        # Retrieve embeddings
        p = self.prompt_emb[idx]
        a = self.res_a_emb[idx]
        b = self.res_b_emb[idx]

        # Compute interaction features
        diff = a - b
        prod = a * b

        # Concatenate all features: [Prompt, ResA, ResB, Diff, Prod]
        # Shape: (EMBEDDING_DIM * 5, )
        features = np.concatenate([p, a, b, diff, prod], axis=0)

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if self.targets is not None:
            target = self.targets[idx]
            return features_tensor, torch.tensor(target, dtype=torch.long)
        else:
            return features_tensor


def get_embeddings_for_split(df, split_name, vectorizer, load_cached_data):
    """
    Handles caching logic: loads embeddings from disk if available/requested,
    otherwise computes them and saves to disk.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{split_name}_embeddings.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings for {split_name} from {cache_file}")
        try:
            data = np.load(cache_file)
            return data["prompt"], data["res_a"], data["res_b"]
        except Exception as e:
            logger.warning(
                f"Failed to load cache for {split_name}: {e}. Recomputing..."
            )

    # Compute embeddings
    logger.info(f"Computing embeddings for {split_name}...")
    prompt_emb = vectorizer.encode(df["prompt"], show_progress_bar=False)
    res_a_emb = vectorizer.encode(df["response_a"], show_progress_bar=False)
    res_b_emb = vectorizer.encode(df["response_b"], show_progress_bar=False)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez(cache_file, prompt=prompt_emb, res_a=res_a_emb, res_b=res_b_emb)
    logger.info(f"Saved embeddings for {split_name} to {cache_file}")

    return prompt_emb, res_a_emb, res_b_emb


def create_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    """
    logger.info("Starting data processing...")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Debug Mode: Subsample
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode enabled. Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows."
        )
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Initialize Vectorizer
    # We only need the vectorizer if we are NOT loading everything from cache,
    # but since we process splits sequentially, we initialize it once.
    # To save memory, we could initialize it inside get_embeddings_for_split only if needed,
    # but keeping it here is cleaner.
    vectorizer = TextVectorizer()

    # 3. Process Splits

    # --- Train ---
    t_p, t_a, t_b = get_embeddings_for_split(
        train_df, "train", vectorizer, load_cached_data
    )
    # Convert one-hot targets to class indices
    # 0: Model A, 1: Model B, 2: Tie
    # Assuming columns: winner_model_a, winner_model_b, winner_tie
    train_targets = np.argmax(
        train_df[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1
    )

    train_dataset = ChatbotDataset(t_p, t_a, t_b, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Validation ---
    v_p, v_a, v_b = get_embeddings_for_split(
        val_df, "val", vectorizer, load_cached_data
    )
    val_targets = np.argmax(
        val_df[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1
    )

    val_dataset = ChatbotDataset(v_p, v_a, v_b, val_targets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    test_p, test_a, test_b = get_embeddings_for_split(
        test_df, "test", vectorizer, load_cached_data
    )
    # Test set has no targets
    test_dataset = ChatbotDataset(test_p, test_a, test_b, targets=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader
