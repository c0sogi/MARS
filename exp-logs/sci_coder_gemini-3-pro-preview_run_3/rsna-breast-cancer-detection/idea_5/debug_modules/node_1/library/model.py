import torch
import torch.nn as nn
import timm
from library.config import Config


class TransformerAggregator(nn.Module):
    """
    Aggregates a sequence of feature vectors using a Transformer Encoder with a [CLS] token.
    """

    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        # Project backbone features to transformer embedding dimension
        self.projector = nn.Linear(input_dim, embed_dim)

        # Learnable [CLS] token to capture global context
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer Encoder
        # batch_first=True expects input shape (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Input_Dim).
            mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len).
                                 True indicates padding (to be ignored).
        Returns:
            torch.Tensor: Aggregated embedding of shape (Batch, Embed_Dim).
        """
        b, s, _ = x.shape

        # Project features
        x = self.projector(x)  # (B, S, E)

        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, E)

        # Prepend CLS token to the sequence
        x = torch.cat((cls_tokens, x), dim=1)  # (B, S+1, E)

        # Adjust mask for CLS token
        # The CLS token is never padded, so we prepend False (do not mask)
        if mask is not None:
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
            mask = torch.cat((cls_mask, mask), dim=1)  # (B, S+1)

        # Transformer Forward
        # src_key_padding_mask: True values are ignored
        out = self.transformer(x, src_key_padding_mask=mask)

        # Return the output corresponding to the [CLS] token (index 0)
        return out[:, 0, :]


class MilTransformerModel(nn.Module):
    """
    Multi-Instance Learning Model for Breast Cancer Detection.

    Architecture:
    1. Backbone (EfficientNetV2): Extracts features from each image in the bag.
    2. Transformer Aggregator: Aggregates features from multiple views.
    3. Multi-Task Heads: Predicts Cancer (Primary), Density (Aux), and Biopsy (Aux).
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Load pretrained EfficientNetV2-Small
        # num_classes=0 removes the final classifier, returning the pooled feature vector
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )
        backbone_dim = self.backbone.num_features

        # 2. Aggregator
        self.aggregator = TransformerAggregator(
            input_dim=backbone_dim,
            embed_dim=Config.TRANSFORMER_EMBED_DIM,
            num_heads=Config.TRANSFORMER_NUM_HEADS,
            num_layers=Config.TRANSFORMER_NUM_LAYERS,
            dropout=Config.DROPOUT,
        )

        # 3. Heads
        # Primary Task: Cancer Detection (Binary)
        self.cancer_head = nn.Linear(Config.TRANSFORMER_EMBED_DIM, 1)

        # Auxiliary Task: Breast Density (4 classes: A, B, C, D)
        self.density_head = nn.Linear(Config.TRANSFORMER_EMBED_DIM, 4)

        # Auxiliary Task: Biopsy Prediction (Binary)
        self.biopsy_head = nn.Linear(Config.TRANSFORMER_EMBED_DIM, 1)

    def forward(self, images_list):
        """
        Args:
            images_list (List[torch.Tensor]): A list of length Batch_Size.
                Each element is a tensor of shape (Num_Views, C, H, W).
                Num_Views can vary per bag.

        Returns:
            dict: Dictionary containing logits for 'cancer', 'density', and 'biopsy'.
        """
        batch_size = len(images_list)
        device = images_list[0].device

        # --- Step 1: Process all images through Backbone ---
        # Flatten the batch to process all views in parallel
        view_counts = [img.shape[0] for img in images_list]
        max_views = max(view_counts)

        # Concatenate all views: (Total_Views, C, H, W)
        all_images = torch.cat(images_list, dim=0)

        # Extract features: (Total_Views, Backbone_Dim)
        features = self.backbone(all_images)

        # --- Step 2: Prepare Sequence for Transformer ---
        # We need to restructure the flat features back into (Batch, Max_Views, Dim)
        # and create a padding mask.

        padded_features = torch.zeros(
            (batch_size, max_views, features.shape[1]),
            dtype=features.dtype,
            device=device,
        )

        # Mask: True indicates padding (ignore)
        padding_mask = torch.ones(
            (batch_size, max_views), dtype=torch.bool, device=device
        )

        cursor = 0
        for i, count in enumerate(view_counts):
            # Slice features belonging to this bag
            bag_feats = features[cursor : cursor + count]
            cursor += count

            # Place into padded tensor
            padded_features[i, :count, :] = bag_feats

            # Update mask (valid positions set to False)
            padding_mask[i, :count] = False

        # --- Step 3: Aggregation ---
        # (Batch, Embed_Dim)
        embedding = self.aggregator(padded_features, mask=padding_mask)

        # --- Step 4: Multi-Task Prediction ---
        cancer_logits = self.cancer_head(embedding)
        density_logits = self.density_head(embedding)
        biopsy_logits = self.biopsy_head(embedding)

        return {
            "cancer": cancer_logits,
            "density": density_logits,
            "biopsy": biopsy_logits,
        }
