import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    Model class for Sentiment Analysis Span Extraction.
    Uses DeBERTa-v3-large backbone with Multi-Sample Dropout (MSD) head.
    """

    def __init__(self):
        super(TweetModel, self).__init__()
        # Load configuration and model from pre-trained path
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_PATH, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(Config.MODEL_PATH, config=self.config)

        # Multi-Sample Dropout (Internal Ensembling)
        # We use a list of dropout layers with different rates to improve generalization
        self.use_msd = Config.USE_MSD
        if self.use_msd:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(p) for p in Config.MSD_DROPOUT_RATES]
            )
        else:
            # Fallback to standard dropout if MSD is disabled
            self.dropouts = nn.ModuleList([nn.Dropout(0.1)])

        # Shared Linear Layer for prediction
        # Maps hidden_size -> 2 (start_logit, end_logit)
        # This layer is shared across all dropout paths
        self.fc = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the new head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the linear layer using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, ids, mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            ids (torch.Tensor): Input token IDs.
            mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.

        Returns:
            start_logits (torch.Tensor): Logits for start position.
            end_logits (torch.Tensor): Logits for end position.
        """
        # Pass inputs through the backbone
        # DeBERTa-v3 accepts token_type_ids (though often not used for single seq,
        # here we have sentiment+text, so it might be useful)
        outputs = self.backbone(
            input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        # Get the sequence of hidden states (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply Multi-Sample Dropout
        start_logits_list = []
        end_logits_list = []

        for dropout in self.dropouts:
            # Apply dropout
            x = dropout(sequence_output)

            # Pass through shared linear layer
            logits = self.fc(x)  # (batch_size, seq_len, 2)

            # Split into start and end logits
            start, end = logits.split(1, dim=-1)

            start_logits_list.append(start.squeeze(-1))
            end_logits_list.append(end.squeeze(-1))

        # Average the predictions from all dropout masks (Internal Ensembling)
        if len(self.dropouts) > 1:
            start_logits = torch.stack(start_logits_list).mean(dim=0)
            end_logits = torch.stack(end_logits_list).mean(dim=0)
        else:
            start_logits = start_logits_list[0]
            end_logits = end_logits_list[0]

        return start_logits, end_logits
