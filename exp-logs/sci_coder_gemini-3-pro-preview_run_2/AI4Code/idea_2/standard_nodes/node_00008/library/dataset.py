import torch
from torch.utils.data import Dataset
from library.config import MAX_LEN


class MarkdownRankDataset(Dataset):
    """
    PyTorch Dataset for the Markdown Ranking task.

    This dataset processes pairs of (Markdown Content, Code Context) and prepares
    them for input into a Transformer model. It handles both training data (with ranks)
    and test data (without ranks).
    """

    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'text', 'context', and optionally 'rank'.
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
            max_len (int): Maximum sequence length for tokenization.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Check if the target column exists to distinguish between Train/Val and Test mode
        self.has_label = "rank" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Retrieve text and context, ensuring they are strings
        text = str(row["text"])
        context = str(row["context"])

        # Tokenize the pair.
        # The tokenizer handles the special tokens for sentence pairs automatically.
        # e.g., for RoBERTa: <s> text </s> </s> context </s>
        inputs = self.tokenizer(
            text,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze to remove the batch dimension added by return_tensors='pt' (1, seq_len) -> (seq_len)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        # If training/validation, include the target rank
        if self.has_label:
            # Convert rank to float tensor
            item["label"] = torch.tensor(row["rank"], dtype=torch.float)

        return item
