import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    TweetModel with a DeBERTa-v3-large backbone and a Multi-Sample Dropout head.

    This model extracts the start and end indices of the sentiment-supporting phrase
    from the input tweet. It leverages Multi-Sample Dropout to improve generalization
    and stability during fine-tuning.
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load configuration and pre-trained backbone
        self.config = AutoConfig.from_pretrained(
            Config.model_name, output_hidden_states=True
        )
        self.bert = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Multi-Sample Dropout: A list of dropout layers with different probabilities
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.msd_dropout_rates])

        # Shared output layer: Projects hidden size to 2 values (start_logit, end_logit)
        self.fc = nn.Linear(self.config.hidden_size, 2)

        # Initialize the weights of the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the specific module using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (segment IDs).

        Returns:
            start_logits (torch.Tensor): Logits for the start index (Batch, Seq_Len).
            end_logits (torch.Tensor): Logits for the end index (Batch, Seq_Len).
        """
        # Pass inputs through the backbone
        # DeBERTa-v3 accepts token_type_ids, though they are less critical than in BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Retrieve the last hidden state: (Batch_Size, Seq_Len, Hidden_Size)
        last_hidden_state = outputs.last_hidden_state

        start_logits_list = []
        end_logits_list = []

        # Apply Multi-Sample Dropout
        # Pass the hidden state through each dropout layer and then the shared linear layer
        for dropout in self.dropouts:
            x = dropout(last_hidden_state)
            logits = self.fc(x)  # (Batch_Size, Seq_Len, 2)

            # Split into start and end logits
            start, end = logits.split(1, dim=-1)

            # Squeeze the last dimension to get (Batch_Size, Seq_Len)
            start_logits_list.append(start.squeeze(-1))
            end_logits_list.append(end.squeeze(-1))

        # Average the predictions from all dropout branches
        start_logits = torch.stack(start_logits_list).mean(dim=0)
        end_logits = torch.stack(end_logits_list).mean(dim=0)

        return start_logits, end_logits
