import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualStreamTextCNN(nn.Module):
    """
    Dual-Stream TextCNN for Stack Exchange Tag Prediction.

    Architecture:
    1. Shared Embedding Layer: Maps token IDs to dense vectors for both Title and Body.
    2. Title Stream: 1D-CNNs with smaller kernels to capture keywords/phrases in Title.
    3. Body Stream: 1D-CNNs with larger kernels to capture context in Body.
    4. Feature Fusion: Concatenation of Global Max Pooled vectors from both streams.
    5. Classifier: Fully Connected layer with Dropout.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes (tags).
        """
        super(DualStreamTextCNN, self).__init__()

        # Load hyperparameters from Config
        vocab_size = Config.VOCAB_SIZE
        embed_dim = Config.EMBED_DIM
        title_kernels = Config.TITLE_KERNELS
        body_kernels = Config.BODY_KERNELS
        num_filters = Config.NUM_FILTERS
        dropout_p = Config.DROPOUT

        # 1. Shared Embedding Layer
        # padding_idx=0 ensures the padding token vector remains zero and is not updated
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. Title Stream Convolutions
        # Captures short, high-signal patterns
        self.title_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                )
                for k in title_kernels
            ]
        )

        # 3. Body Stream Convolutions
        # Captures broader context
        self.body_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                )
                for k in body_kernels
            ]
        )

        # Calculate input dimension for the fully connected layer
        # Sum of filters from all kernels in both streams
        total_filters = (len(title_kernels) + len(body_kernels)) * num_filters

        # 4. Classifier
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(total_filters, num_classes)

    def forward_stream(self, x, convs):
        """
        Applies convolution, non-linearity, and global max pooling for a specific stream.

        Args:
            x (torch.Tensor): Embedded input tensor of shape (Batch, Embed_Dim, Seq_Len).
            convs (nn.ModuleList): List of Conv1d layers.

        Returns:
            torch.Tensor: Concatenated pooled features for this stream.
        """
        # Apply Conv1d -> ReLU -> Global Max Pool
        # Output of conv(x): (Batch, Num_Filters, Seq_Len - Kernel + 1)
        # Output of max_pool1d: (Batch, Num_Filters, 1)
        # Squeeze: (Batch, Num_Filters)

        features = [
            F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).shape[2]).squeeze(2)
            for conv in convs
        ]

        # Concatenate features from different kernel sizes
        return torch.cat(features, dim=1)

    def forward(self, title, body):
        """
        Forward pass of the model.

        Args:
            title (torch.Tensor): Title indices (Batch, Title_Len).
            body (torch.Tensor): Body indices (Batch, Body_Len).

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # 1. Embedding
        # Shape: (Batch, Seq_Len, Embed_Dim)
        x_title = self.embedding(title)
        x_body = self.embedding(body)

        # 2. Permute for Conv1d
        # Conv1d expects (Batch, Channels, Length)
        x_title = x_title.permute(0, 2, 1)
        x_body = x_body.permute(0, 2, 1)

        # 3. Process Streams
        enc_title = self.forward_stream(x_title, self.title_convs)
        enc_body = self.forward_stream(x_body, self.body_convs)

        # 4. Feature Fusion
        combined = torch.cat((enc_title, enc_body), dim=1)

        # 5. Classification
        out = self.dropout(combined)
        logits = self.fc(out)

        return logits
