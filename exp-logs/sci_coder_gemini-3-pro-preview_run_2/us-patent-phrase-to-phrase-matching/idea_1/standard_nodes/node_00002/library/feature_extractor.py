import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, logging
from library.config import Config

# Suppress verbose warnings from transformers
logging.set_verbosity_error()


class SentenceEncoder:
    """
    A wrapper around a pre-trained Transformer model to generate
    dense vector representations (embeddings) for text inputs.
    """

    def __init__(self):
        """
        Initializes the tokenizer and model based on the configuration.
        Moves the model to the specified device and sets it to evaluation mode.
        """
        self.device = Config.DEVICE
        self.model_name = Config.MODEL_NAME

        # Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Move to device and set to eval mode (Frozen)
        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the token embeddings, taking the attention mask into account.

        Args:
            model_output: The output from the transformer model.
            attention_mask: The attention mask tensor.

        Returns:
            torch.Tensor: The pooled sentence embeddings.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def encode(self, texts, batch_size=None):
        """
        Generates embeddings for a list of texts.

        Args:
            texts (list[str]): A list of text strings to encode.
            batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.

        Returns:
            np.ndarray: A numpy array of shape (n_samples, embedding_dim).
        """
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=Config.MAX_LENGTH,
                return_tensors="pt",
            )

            # Move inputs to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Forward pass
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Mean Pooling
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            # Move to CPU and numpy
            embeddings = sentence_embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if not all_embeddings:
            return np.array([])

        return np.concatenate(all_embeddings, axis=0)


def extract_embeddings(df, encoder, split, load_cached_data=True):
    """
    Extracts embeddings for the 'anchor_input' and 'target_input' columns of the dataframe.
    Implements caching logic to save/load embeddings to/from disk.

    Args:
        df (pd.DataFrame): The dataframe containing text data.
        encoder (SentenceEncoder): The initialized encoder instance.
        split (str): The dataset split name ('train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (anchor_embeddings, target_embeddings) as numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Construct cache filename
    # Append _debug suffix if running in debug mode to separate cache files
    suffix = "_debug" if Config.DEBUG else ""
    filename = f"{split}_embeddings{suffix}.npz"
    cache_path = os.path.join(Config.CACHE_DIR, filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached embeddings for {split} from {cache_path}")
        try:
            data = np.load(cache_path)
            return data["anchors"], data["targets"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    # print(f"Computing embeddings for {split}...")

    # Extract text lists
    # Ensure data is string type
    anchors = df["anchor_input"].astype(str).tolist()
    targets = df["target_input"].astype(str).tolist()

    # Encode
    anchor_embeddings = encoder.encode(anchors)
    target_embeddings = encoder.encode(targets)

    # 3. Save to cache
    np.savez(cache_path, anchors=anchor_embeddings, targets=target_embeddings)
    # print(f"Saved embeddings for {split} to {cache_path}")

    return anchor_embeddings, target_embeddings
