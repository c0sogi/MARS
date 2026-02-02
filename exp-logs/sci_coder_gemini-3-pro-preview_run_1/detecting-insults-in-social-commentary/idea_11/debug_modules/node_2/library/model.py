import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridModel(nn.Module):
    """
    Heterogeneous Hybrid Stacking Architecture Model.
    Combines a Transformer backbone (DeBERTa-v3 or RoBERTa) with SVD-based structural features.

    Architecture:
    1. Transformer Backbone -> [CLS] Embedding
    2. SVD Features -> Linear -> LayerNorm -> GELU -> SVD Embedding
    3. Concatenation ([CLS], SVD Embedding)
    4. Variable-Rate Multi-Sample Dropout (VR-MSD) -> Classifier -> Logits
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): HuggingFace model name (e.g., 'microsoft/deberta-v3-large').
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing for memory efficiency
        # Essential for training Large models on limited VRAM
        # Cite debug_lesson_7: Explicitly disable reentrant checkpointing to avoid backward pass errors
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Feature Dimensions
        self.hidden_size = self.config.hidden_size
        self.svd_input_dim = Config.svd_components
        self.svd_projected_dim = 256  # Projecting SVD features to a dense vector

        # SVD Fusion Head
        # Projects sparse-derived SVD features into a dense, normalized embedding
        self.svd_head = nn.Sequential(
            nn.Linear(self.svd_input_dim, self.svd_projected_dim),
            nn.LayerNorm(self.svd_projected_dim),
            nn.GELU(),
        )

        # Variable-Rate Multi-Sample Dropout (VR-MSD)
        # Uses multiple dropout rates to smooth the loss landscape
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.dropout_rates])

        # Final Classifier
        # Input: Concatenation of [CLS] embedding and SVD projected vector
        self.fc = nn.Linear(self.hidden_size + self.svd_projected_dim, 1)

        # Initialize weights for the new heads
        self._init_weights(self.svd_head)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom layers (SVD head and Classifier).
        Follows standard transformer initialization patterns.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for submodule in module:
                self._init_weights(submodule)

    def forward(self, input_ids, attention_mask, svd_features, token_type_ids=None):
        """
        Forward pass of the Hybrid Model.

        Args:
            input_ids (torch.Tensor): Token indices.
            attention_mask (torch.Tensor): Attention mask.
            svd_features (torch.Tensor): Pre-computed SVD features.
            token_type_ids (torch.Tensor, optional): Token type IDs (if supported by backbone).

        Returns:
            torch.Tensor: Logits (averaged across MSD heads).
        """
        # Prepare inputs for the backbone
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Include token_type_ids if provided (e.g., for DeBERTa, or if dataset generates them)
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # 1. Backbone Forward
        outputs = self.backbone(**inputs)

        # Extract [CLS] token representation
        # For both DeBERTa and RoBERTa, this is the first token of the last hidden state
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # 2. SVD Feature Processing
        svd_embedding = self.svd_head(svd_features)

        # 3. Feature Fusion
        # Concatenate the semantic (Transformer) and structural (SVD) embeddings
        fused_features = torch.cat([cls_embedding, svd_embedding], dim=1)

        # 4. Multi-Sample Dropout & Classification
        # Pass the fused features through multiple dropout layers and the same classifier
        logits_list = []
        for dropout in self.dropouts:
            dropped = dropout(fused_features)
            logits_list.append(self.fc(dropped))

        # Average the predictions (logits) from all dropout samples
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits
