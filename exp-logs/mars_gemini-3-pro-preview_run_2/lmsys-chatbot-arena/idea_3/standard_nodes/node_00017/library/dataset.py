import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Chatbot Preference task.

    Handles the tokenization of (Prompt, Response A) and (Prompt, Response B) pairs
    and extracts pre-calculated meta-features.
    """

    def __init__(
        self, df, tokenizer, max_length=Config.MAX_LENGTH, is_test=False, scaler=None
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing text and meta-feature columns.
                               Expected columns: 'prompt', 'response_a', 'response_b',
                               'meta_prompt_len', 'meta_a_len', 'meta_b_len'.
                               If is_test=False, also 'winner_model_a', 'winner_model_b', 'winner_tie'.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            is_test (bool): Whether this is the test set (no targets).
            scaler: Optional scaler object (included for signature compliance,
                    though data is assumed to be pre-scaled in data_processor).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.scaler = scaler

        # Convert columns to numpy arrays for faster access than pandas .iloc
        self.prompts = df["prompt"].values.astype(str)
        self.response_a = df["response_a"].values.astype(str)
        self.response_b = df["response_b"].values.astype(str)

        # Meta features: ['meta_prompt_len', 'meta_a_len', 'meta_b_len']
        # These are assumed to be already scaled by data_processor.py
        self.meta_features = df[
            ["meta_prompt_len", "meta_a_len", "meta_b_len"]
        ].values.astype(np.float32)

        if not self.is_test:
            # Targets: ['winner_model_a', 'winner_model_b', 'winner_tie']
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.response_a[idx]
        resp_b = self.response_b[idx]

        # Tokenize (Prompt, Response A)
        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize (Prompt, Response B)
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract Meta Features
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float)

        # Prepare Output Dictionary
        # Squeeze(0) is needed because return_tensors='pt' adds a batch dimension (1, seq_len)
        item = {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "meta_features": meta,
        }

        if not self.is_test:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            item["target"] = target

        return item
