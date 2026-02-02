import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from library.config import Config


class JigsawTransformer(nn.Module):
    """
    Transformer-based model for Jigsaw Toxicity Classification.

    Architecture:
    1. Backbone: Pre-trained RoBERTa model (e.g., roberta-base).
    2. Adaptation: Low-Rank Adaptation (LoRA) applied to attention projections to enable
       parameter-efficient fine-tuning.
    3. Heads: Two decoupled linear heads:
       - Toxicity Head: Predicts the main toxicity score.
       - Identity Head: Predicts identity mentions (used for bias mitigation loss).
    """

    def __init__(self):
        super(JigsawTransformer, self).__init__()

        # 1. Load Configuration
        # Retrieve hidden size and dropout settings from the pre-trained config
        hf_config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.hidden_size = hf_config.hidden_size

        # 2. Load Base Model
        base_model = AutoModel.from_pretrained(Config.MODEL_NAME)

        # 3. Apply LoRA (Low-Rank Adaptation)
        # This wraps the base model, freezing original weights and adding trainable adapters
        if Config.USE_LORA:
            peft_config = LoraConfig(
                r=Config.LORA_R,
                lora_alpha=Config.LORA_ALPHA,
                target_modules=Config.LORA_TARGET_MODULES,
                lora_dropout=Config.LORA_DROPOUT,
                bias="none",
                task_type=None,  # Using as generic feature extractor backbone
            )
            self.backbone = get_peft_model(base_model, peft_config)

            # Print summary of trainable parameters to verify LoRA setup
            print("Initialized LoRA Backbone:")
            self.backbone.print_trainable_parameters()
        else:
            self.backbone = base_model

        # 4. Define Heads
        # Dropout applied to the pooled output before classification
        dropout_prob = getattr(hf_config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # Toxicity Head: Projects [CLS] embedding to a single scalar (logit)
        self.toxicity_head = nn.Linear(self.hidden_size, 1)

        # Identity Head: Projects [CLS] embedding to identity attributes
        # Used for the auxiliary loss to disentangle toxicity from identity
        num_identities = len(Config.IDENTITY_COLS)
        self.identity_head = nn.Linear(self.hidden_size, num_identities)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the neural network.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                                           Shape: (batch_size, sequence_length)

        Returns:
            tuple: (toxicity_logits, identity_logits)
        """
        # Pass inputs through the Transformer backbone
        # The peft_model forwards arguments to the underlying base model
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token representation
        # In RoBERTa, this is the first token (index 0) of the last hidden state
        # Shape: (batch_size, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Apply regularization
        features = self.dropout(cls_token)

        # Compute predictions for both tasks
        toxicity_logits = self.toxicity_head(features)
        identity_logits = self.identity_head(features)

        return toxicity_logits, identity_logits
