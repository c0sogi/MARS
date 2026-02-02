import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class HybridDebertaModel(nn.Module):
    """
    Hybrid DeBERTa-v3 model with Structural Feature Fusion.

    This architecture integrates deep semantic representations from DeBERTa-v3
    with explicit structural features (SVD-compressed character N-grams) to
    enhance detection of insults, particularly those using obfuscation.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_structural_features=Config.SVD_COMPONENTS,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Initializes the Hybrid DeBERTa Model.

        Args:
            model_name (str): Name of the pre-trained Transformer model.
            num_structural_features (int): Size of the auxiliary structural feature vector.
            hidden_size (int): Hidden size of the Transformer's output embeddings.
            dropout_rate (float): Dropout probability for the fusion head.
        """
        super(HybridDebertaModel, self).__init__()

        # Load the pre-trained backbone
        self.backbone = AutoModel.from_pretrained(model_name)

        # Calculate input dimension for the fusion layer
        # Combination of [CLS] embedding and structural features
        fusion_input_dim = hidden_size + num_structural_features

        # Define the Fusion Classification Head
        # Architecture: Dropout -> Linear -> ReLU -> Linear -> Logit
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask, structural_features):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer. Shape: (B, Seq_Len)
            attention_mask (torch.Tensor): Attention masks. Shape: (B, Seq_Len)
            structural_features (torch.Tensor): Dense structural features. Shape: (B, Num_Struct_Feats)

        Returns:
            torch.Tensor: Raw logits for binary classification. Shape: (B, 1)
        """
        # 1. Semantic Branch: Process text through DeBERTa
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token embedding (index 0)
        # Shape: (Batch_Size, Hidden_Size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 2. Feature Fusion: Concatenate semantic and structural signals
        # Shape: (Batch_Size, Hidden_Size + Num_Struct_Feats)
        fused_vector = torch.cat((cls_embedding, structural_features), dim=1)

        # 3. Classification Head: Generate logits
        logits = self.fusion_head(fused_vector)

        return logits
