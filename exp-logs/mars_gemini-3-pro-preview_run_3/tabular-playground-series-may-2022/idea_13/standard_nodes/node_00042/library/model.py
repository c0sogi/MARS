import numpy as np
import torch
import torch.nn as nn


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP implementation with Entity Embeddings.
    Uses ReLU activations and standard Dropout.
    """

    def __init__(self, num_cont, vocab_sizes, embed_dim, hidden_layers, dropout_rate):
        super(ManufacturingMLP, self).__init__()

        # 1. Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # 2. Backbone Construction
        input_dim = num_cont + (len(vocab_sizes) * embed_dim)
        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Prediction Head
        self.head = nn.Linear(in_dim, 1)

        # 4. Initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_cont, x_cat):
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded_features.append(emb_layer(x_cat[:, i]))

        x_emb = torch.cat(embedded_features, dim=1)
        x = torch.cat([x_cont, x_emb], dim=1)
        x = self.backbone(x)
        logits = self.head(x)
        return logits
