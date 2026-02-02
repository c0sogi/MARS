import torch
import torch.nn as nn
import timm
from library.config import Config


class SiameseMultiScaleDiffNet(nn.Module):
    """
    Siamese EfficientNet-B0 with Enhanced Pooling.

    Updated based on Lesson 00026 and Lesson 00027:
    - Uses only the deepest feature map (Single Scale).
    - Applies both GAP and GMP to On-Target, Off-Target, and Difference streams.
    """

    def __init__(self):
        super(SiameseMultiScaleDiffNet, self).__init__()

        # Initialize the backbone with features_only=True
        # out_indices=(4,) targets the last stage (Stride 32)
        # Cite solution_lesson_node_00026: Prefer Deepest-Layer Feature Comparison
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(4,),
            in_chans=Config.IN_CHANNELS,
        )

        # Determine the input dimension for the final linear layer
        dummy_input = torch.zeros(
            1, Config.IN_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
        )
        with torch.no_grad():
            # features is a list containing the single feature map
            features = self.backbone(dummy_input)

        # Calculate total embedding dimension
        # We extract 6 vectors from the single scale:
        # 1. GAP(On), 2. GMP(On)
        # 3. GAP(Off), 4. GMP(Off)
        # 5. GAP(Diff), 6. GMP(Diff)
        # Cite solution_lesson_node_00027: Source-Stream Max Pooling
        self.embed_dim = features[0].shape[1] * 6

        # Single linear classification layer
        # Cite solution_lesson_node_00021: Simplify Classification Heads
        self.fc = nn.Linear(self.embed_dim, 1)

    def forward_features(self, x):
        """
        Passes input through the backbone and returns the deepest feature map.
        """
        return self.backbone(x)[0]

    def forward(self, stream_a, stream_b):
        """
        Forward pass for the Siamese Network.
        """
        # Extract features for both streams
        fa = self.forward_features(stream_a)
        fb = self.forward_features(stream_b)

        # Compute Explicit Spatial Difference
        # Cite solution_lesson_node_00022: Spatial Feature Subtraction
        f_diff = fa - fb

        pooled_vectors = []

        # Apply GAP and GMP to all streams (On, Off, Diff)
        # Cite solution_lesson_node_00027
        for f in [fa, fb, f_diff]:
            pooled_vectors.append(f.mean(dim=(2, 3)))  # GAP
            pooled_vectors.append(f.amax(dim=(2, 3)))  # GMP

        # Concatenate all vectors
        concat_features = torch.cat(pooled_vectors, dim=1)

        # Final classification
        logits = self.fc(concat_features)

        return logits
