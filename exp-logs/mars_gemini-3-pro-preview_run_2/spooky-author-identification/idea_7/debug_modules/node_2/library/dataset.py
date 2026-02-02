import torch
from torch.utils.data import Dataset
from library.config import Config


class AuthorDataset(Dataset):
    """
    A custom Dataset class for handling text data for the Author Identification task.
    It handles tokenization using a provided HuggingFace tokenizer and prepares
    tensors for the model.
    """

    def __init__(self, texts, labels, tokenizer, max_length=Config.MAX_LENGTH):
        """
        Args:
            texts (list or np.ndarray): Sequence of text strings to classify.
            labels (list or np.ndarray, optional): Sequence of integer labels corresponding to authors.
                                                   Pass None for test/inference sets.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer instance to use.
            max_length (int): Maximum sequence length for tokenization. Defaults to Config.MAX_LENGTH.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index, tokenizes it, and returns the necessary tensors.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'input_ids': Tensor of token IDs.
                - 'attention_mask': Tensor indicating non-padded tokens.
                - 'target': Tensor of the label (if labels were provided).
        """
        # Ensure text is a string
        text = str(self.texts[idx])

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Pad & truncate to max_length
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # The tokenizer returns tensors with shape (1, max_length), so we flatten them
        # to shape (max_length,) for the DataLoader to stack correctly.
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Include label if available
        if self.labels is not None:
            # Convert label to a LongTensor (standard for CrossEntropyLoss)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            item["target"] = label

        return item
