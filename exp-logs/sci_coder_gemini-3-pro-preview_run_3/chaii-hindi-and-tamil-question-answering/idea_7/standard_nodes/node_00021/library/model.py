import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoConfig
from library.config import Config


def get_class_weights(dataset):
    """
    Calculates inverse frequency weights for the classes based on the dataset.

    Args:
        dataset: A PyTorch Dataset (specifically QADataset) or an iterable of samples.
                 If it has a 'df_features' attribute (pandas DataFrame), a fast path is used.

    Returns:
        torch.Tensor: A tensor of shape (num_labels,) containing the weights.
    """
    print("Calculating class weights from training data...")

    # Fast path: if dataset has the underlying dataframe exposed
    if hasattr(dataset, "df_features"):
        # Concatenate all label lists to count efficiently
        all_labels = np.concatenate(dataset.df_features["labels"].values)
        unique, counts = np.unique(all_labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
    else:
        # Slow path: iterate over the dataset
        count_dict = {}
        for i in range(len(dataset)):
            item = dataset[i]
            labels = item["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()

            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                count_dict[u] = count_dict.get(u, 0) + c

    # Calculate weights: Total / (Num_Classes * Class_Count)
    total_count = sum(count_dict.values())
    num_classes = Config.NUM_LABELS
    weights = []

    for i in range(num_classes):
        count = count_dict.get(i, 0)
        if count > 0:
            w = total_count / (num_classes * count)
        else:
            w = 1.0  # Default if class is missing (unlikely)
        weights.append(w)

    print(f"Class counts: {count_dict}")
    print(f"Computed weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


class WeightedTokenClassifier(nn.Module):
    """
    XLM-Roberta based Token Classifier with Class-Weighted Cross-Entropy Loss.
    """

    def __init__(self, class_weights=None):
        """
        Args:
            class_weights (torch.Tensor or list, optional): Weights for each class to handle imbalance.
        """
        super(WeightedTokenClassifier, self).__init__()

        self.config = AutoConfig.from_pretrained(Config.MODEL_CHECKPOINT)
        self.roberta = AutoModel.from_pretrained(
            Config.MODEL_CHECKPOINT, config=self.config
        )

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, Config.NUM_LABELS)

        # Initialize class weights
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.class_weights = class_weights
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).
            labels (torch.Tensor, optional): Ground truth labels (batch_size, seq_len).

        Returns:
            tuple: (loss, logits) if labels are provided.
            torch.Tensor: logits if labels are not provided.
        """
        # Pass through RoBERTa
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            # Ensure weights are on the correct device
            if self.class_weights is not None:
                if self.class_weights.device != logits.device:
                    self.class_weights = self.class_weights.to(logits.device)

            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

            # Only compute loss on active tokens (where attention_mask == 1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, Config.NUM_LABELS)
                active_labels = labels.view(-1)

                # Filter by attention mask
                loss = loss_fct(active_logits[active_loss], active_labels[active_loss])
            else:
                loss = loss_fct(logits.view(-1, Config.NUM_LABELS), labels.view(-1))

        if loss is not None:
            return loss, logits
        return logits
