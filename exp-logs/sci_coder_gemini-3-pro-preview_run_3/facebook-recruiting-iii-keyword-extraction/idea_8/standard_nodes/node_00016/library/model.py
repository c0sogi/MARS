import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Down-weights easy examples to focus training on hard positives/negatives.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the loss (default: 1.0).
            gamma (float): Focusing parameter (default: 2.0).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (N, C).
            targets (torch.Tensor): Binary targets (N, C).
        """
        # Compute binary cross entropy loss (numerically stable with logits)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # pt is the probability of the true class
        # Since bce_loss = -log(pt), we can just take exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Compute Focal Loss
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WideAndDeepTextCNN(nn.Module):
    """
    Word-Level Wide-and-Deep TextCNN with Shared Embeddings.

    Structure:
    1. Shared Embedding: Maps word IDs to dense vectors.
    2. Wide Stream: Global Sum Pooling -> Linear. Captures keyword-tag associations (Memorization).
    3. Deep Stream: TextCNN (Conv1d -> MaxPool) -> Linear. Captures n-gram contexts (Generalization).
    4. Fusion: Element-wise sum of logits from Wide and Deep streams.
    """

    def __init__(
        self,
        vocab_size=None,
        embed_dim=None,
        num_tags=None,
        cnn_filters=None,
        cnn_kernel_sizes=None,
        dropout=None,
    ):
        super(WideAndDeepTextCNN, self).__init__()

        # Load defaults from Config if not provided
        self.vocab_size = vocab_size if vocab_size is not None else Config.VOCAB_SIZE
        self.embed_dim = embed_dim if embed_dim is not None else Config.EMBED_DIM
        self.num_tags = num_tags if num_tags is not None else Config.NUM_TAGS
        self.cnn_filters = (
            cnn_filters if cnn_filters is not None else Config.CNN_FILTERS
        )
        self.cnn_kernel_sizes = (
            cnn_kernel_sizes
            if cnn_kernel_sizes is not None
            else Config.CNN_KERNEL_SIZES
        )
        self.dropout_prob = dropout if dropout is not None else Config.DROPOUT

        # 1. Shared Embedding Layer
        # padding_idx=0 ensures the padding token vector is always zero,
        # which is crucial for the Sum Pooling in the Wide stream.
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # 2. Wide Stream (Memorization)
        # Projects the summed embedding directly to the output space
        self.wide_linear = nn.Linear(self.embed_dim, self.num_tags)

        # 3. Deep Stream (Generalization - TextCNN)
        # Parallel 1D Convolutions with different kernel sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.cnn_filters,
                    kernel_size=k,
                )
                for k in self.cnn_kernel_sizes
            ]
        )

        # Output dimension of CNN part is (Number of Kernels * Filters per Kernel)
        deep_output_dim = len(self.cnn_kernel_sizes) * self.cnn_filters
        self.deep_linear = nn.Linear(deep_output_dim, self.num_tags)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input_ids):
        """
        Args:
            input_ids (torch.Tensor): Input sequence indices (Batch, Max_Len).

        Returns:
            torch.Tensor: Logits (Batch, Num_Tags).
        """
        # Shared Embedding
        # Shape: (Batch, Max_Len, Embed_Dim)
        embeds = self.embedding(input_ids)

        # ---------------------------
        # Wide Stream
        # ---------------------------
        # Global Sum Pooling: Sum embeddings across the sequence dimension
        # Shape: (Batch, Embed_Dim)
        wide_pool = torch.sum(embeds, dim=1)

        # Linear projection to tag space
        # Shape: (Batch, Num_Tags)
        wide_logits = self.wide_linear(wide_pool)

        # ---------------------------
        # Deep Stream
        # ---------------------------
        # Permute for Conv1d: (Batch, Embed_Dim, Max_Len)
        deep_in = embeds.permute(0, 2, 1)

        conv_outs = []
        for conv in self.convs:
            # Apply Conv1d
            # Shape: (Batch, Filters, L_out)
            c = conv(deep_in)

            # Apply ReLU activation
            c = F.relu(c)

            # Global Max Pooling over time
            # Shape: (Batch, Filters)
            c = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            conv_outs.append(c)

        # Concatenate outputs from all kernels
        # Shape: (Batch, Filters * Num_Kernels)
        deep_pool = torch.cat(conv_outs, dim=1)

        # Apply Dropout and Linear projection
        deep_pool = self.dropout(deep_pool)
        deep_logits = self.deep_linear(deep_pool)

        # ---------------------------
        # Fusion
        # ---------------------------
        # Sum logits from both streams
        # Shape: (Batch, Num_Tags)
        logits = wide_logits + deep_logits

        return logits
