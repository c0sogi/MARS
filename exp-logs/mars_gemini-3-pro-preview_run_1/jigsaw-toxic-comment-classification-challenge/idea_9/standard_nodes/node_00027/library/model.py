import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class LinearAttentionPooling(nn.Module):
    """
    Linear Attention Pooling:
    Computes a weighted average of the hidden states using a learned attention mechanism.
    Formula: weights = softmax(W * h_t), context = sum(weights * h_t)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Calculate attention scores
        # (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens by setting them to a large negative value
        # attention_mask is 1 for tokens, 0 for padding
        # We expand mask to (batch_size, seq_len, 1)
        mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        w = w + mask

        # Softmax over the sequence length dimension
        weights = torch.softmax(w, dim=1)

        # Weighted sum of hidden states
        # (batch_size, hidden_size)
        context = torch.sum(weights * last_hidden_state, dim=1)

        return context


class CustomDeberta(nn.Module):
    def __init__(self, pretrained=True, checkpoint_path=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Initialize Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(Config.model_name)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Hybrid Pooling Head
        self.pooling_head = LinearAttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout
        # We create a list of dropout layers to be applied in parallel
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.dropout) for _ in range(Config.multi_sample_dropout_num)]
        )

        # Final Classification Layer
        # Input size is hidden_size * 2 because we concat MaxPool and AttnPool
        self.fc = nn.Linear(self.config.hidden_size * 2, Config.num_labels)

        # Initialize weights for the custom head
        self._init_weights(self.pooling_head.attention)
        self._init_weights(self.fc)

        # Load Domain-Adapted Weights if provided
        if checkpoint_path:
            self.load_dapt_weights(checkpoint_path)

    def _init_weights(self, module):
        """
        Standard weight initialization for linear layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def load_dapt_weights(self, path):
        """
        Loads weights from a Domain-Adapted Pre-training (MLM) checkpoint.
        Handles both directory paths (saved via save_pretrained) and direct .pth files.
        """
        print(f"Attempting to load DAPT weights from {path}...")
        try:
            if os.path.isdir(path):
                # If path is a directory, AutoModel handles loading the base model
                # from the MaskedLM checkpoint structure automatically.
                self.backbone = AutoModel.from_pretrained(path)
                print("Successfully loaded DAPT backbone from directory.")
            else:
                # If it's a file, load state dict manually
                state_dict = torch.load(path, map_location="cpu")

                # Filter and adjust keys to match the backbone
                backbone_keys = self.backbone.state_dict().keys()
                new_state_dict = {}

                for k, v in state_dict.items():
                    # Handle 'deberta.' prefix often present in MaskedLM models
                    if k.startswith("deberta."):
                        key_suffix = k[len("deberta.") :]
                        if key_suffix in backbone_keys:
                            new_state_dict[key_suffix] = v
                    elif k in backbone_keys:
                        new_state_dict[k] = v

                if len(new_state_dict) > 0:
                    missing, unexpected = self.backbone.load_state_dict(
                        new_state_dict, strict=False
                    )
                    print(f"Loaded state dict. Missing keys: {len(missing)}")
                else:
                    print("Warning: No matching keys found in checkpoint for backbone.")

        except Exception as e:
            print(f"Failed to load DAPT weights: {e}")
            print("Continuing with standard pre-trained weights.")

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Backbone Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, SeqLen, Hidden)

        # 2. Global Max Pooling
        # Mask padding tokens to -inf so they don't affect max
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        last_hidden_state_masked = last_hidden_state.clone()
        # Set padding areas to very small number
        last_hidden_state_masked[input_mask_expanded == 0] = -1e4

        max_embeddings = torch.max(last_hidden_state_masked, dim=1)[
            0
        ]  # (Batch, Hidden)

        # 3. Linear Attention Pooling
        att_embeddings = self.pooling_head(
            last_hidden_state, attention_mask
        )  # (Batch, Hidden)

        # 4. Concatenate Features
        combined_features = torch.cat(
            [max_embeddings, att_embeddings], dim=1
        )  # (Batch, 2*Hidden)

        # 5. Multi-Sample Dropout & Classification
        # Pass the features through multiple dropout masks and average the logits
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(combined_features)))

        # Average the logits from all dropout paths
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"logits": logits, "loss": loss}
