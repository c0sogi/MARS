import torch
import torch.nn as nn
from library.config import EMBEDDING_DIM, HIDDEN_LAYERS, DROPOUT_RATE


class EntityEmbeddingMLP(nn.Module):
    """
    Feed-Forward Neural Network with Entity Embeddings for mixed categorical and continuous data.
    """

    def __init__(
        self,
        vocab_sizes,
        num_continuous,
        embedding_dim=EMBEDDING_DIM,
        hidden_layers=HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        Args:
            vocab_sizes (list[int]): A list containing the vocabulary size (max index + 1)
                                     for each categorical feature.
            num_continuous (int): The number of continuous input features.
            embedding_dim (int): The dimension of the embedding vector for categorical features.
            hidden_layers (list[int]): A list defining the number of units in each hidden layer.
            dropout_rate (float): The probability of zeroing an element in the dropout layers.
        """
        super(EntityEmbeddingMLP, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.num_continuous = num_continuous

        # Initialize an embedding layer for each categorical feature
        # We use a ModuleList to register these sub-modules properly
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embedding_dim)
                for v in vocab_sizes
            ]
        )

        # Calculate the total dimension of the concatenated input vector
        # Input = [Continuous Features] + [Cat 1 Embedding] + ... + [Cat N Embedding]
        input_dim = num_continuous + (len(vocab_sizes) * embedding_dim)

        # Construct the MLP (Dense -> ReLU -> Dropout) layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Final output layer for binary classification
        self.output_layer = nn.Linear(current_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous_data, categorical_data):
        """
        Forward pass of the model.

        Args:
            continuous_data (torch.Tensor): Tensor of shape (batch_size, num_continuous)
            categorical_data (torch.Tensor): Tensor of shape (batch_size, num_categorical)
                                             containing integer indices.

        Returns:
            torch.Tensor: Probability scores of shape (batch_size, 1).
        """
        # 1. Process Categorical Data
        # Iterate through each categorical feature column and its corresponding embedding layer
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            # Extract the i-th column: shape (batch_size,)
            col_indices = categorical_data[:, i]
            # Lookup embeddings: shape (batch_size, embedding_dim)
            emb = emb_layer(col_indices)
            embedded_features.append(emb)

        # Concatenate all embeddings along the feature dimension
        # Result shape: (batch_size, num_categorical * embedding_dim)
        x_cat = torch.cat(embedded_features, dim=1)

        # 2. Combine with Continuous Data
        # Result shape: (batch_size, total_input_dim)
        x = torch.cat([x_cat, continuous_data], dim=1)

        # 3. Pass through MLP
        x = self.mlp(x)

        # 4. Output Prediction
        x = self.output_layer(x)
        output = self.sigmoid(x)

        return output
