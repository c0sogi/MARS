import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_cpc_texts


def load_and_prepare_data(csv_path: str, load_cached_data: bool = True):
    """
    Loads the dataset from the given CSV path, maps CPC codes to full descriptions,
    and returns a pandas DataFrame. Implements caching to Parquet.

    Args:
        csv_path (str): Path to the source CSV file (e.g., metadata/train.csv).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe with a 'context_text' column.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Construct cache filename based on the input filename
    base_name = os.path.basename(csv_path).replace(".csv", "")
    cache_path = os.path.join(Config.working_dir, f"{base_name}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Basic validation to ensure expected columns exist
            if "context_text" in df.columns:
                return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Retrieve CPC descriptions
    # We propagate the load_cached_data flag to the utils function
    cpc_texts = get_cpc_texts(load_cached_data=load_cached_data)

    # Map context codes to descriptions
    # Default to the code itself if description is missing (though unlikely)
    df["context_text"] = df["context"].map(cpc_texts).fillna(df["context"])

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


class PearsonDataset(Dataset):
    """
    PyTorch Dataset for the Patent Phrase Matching task.
    Implements the Three-Stage Semantic Input strategy.
    """

    def __init__(self, df, tokenizer, max_length=133):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'anchor', 'target', 'context_text', and optionally 'score'.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-extract columns to lists for faster access
        self.anchors = df["anchor"].astype(str).tolist()
        self.targets = df["target"].astype(str).tolist()
        self.contexts = df["context_text"].astype(str).tolist()

        # Check if score exists (it won't for the test set)
        if "score" in df.columns:
            self.scores = df["score"].values.astype("float32")
        else:
            self.scores = None

        # Pre-extract IDs if needed for submission, though usually handled by the caller via the dataframe
        self.ids = df["id"].tolist() if "id" in df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context_text = self.contexts[idx]

        # Strategy: Three-Stage Semantic Input
        # Segment A: Context Description + [SEP] + Anchor Phrase
        # Segment B: Target Phrase

        # We insert the separator manually for the first segment.
        # Note: We add spaces to ensure tokens don't merge unexpectedly.
        sep = self.tokenizer.sep_token
        text_segment_a = f"{context_text} {sep} {anchor}"
        text_segment_b = target

        # Tokenize
        # Passing text and text_pair automatically handles the [SEP] between Segment A and Segment B
        # and generates appropriate token_type_ids.
        inputs = self.tokenizer(
            text_segment_a,
            text_segment_b,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by return_tensors='pt'
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Handle token_type_ids if the tokenizer returns them (DeBERTa V3 usually does)
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"].squeeze(0)
        else:
            # Fallback for tokenizers that don't return type ids (though DeBERTa should)
            token_type_ids = torch.zeros_like(input_ids)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Add label if available
        if self.scores is not None:
            item["label"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item
