import pandas as pd
from library.config import Config


def extract_features(df: pd.DataFrame, split: str):
    """
    Prepares text data for the Transformer model.
    Instead of TF-IDF, we simply return the raw text list.
    Tokenization happens inside the Dataset class.

    Args:
        df (pd.DataFrame): The dataframe containing the text column.
        split (str): The data split ('train', 'val', 'test').

    Returns:
        list: List of text strings.
    """
    # Ensure text column exists
    if Config.TEXT_COL not in df.columns:
        raise KeyError(f"Column '{Config.TEXT_COL}' not found in dataframe.")

    # Fill NaNs with empty string
    texts = df[Config.TEXT_COL].fillna("").astype(str).tolist()

    # We no longer need to save/load a vectorizer as we use a pretrained tokenizer
    return texts
