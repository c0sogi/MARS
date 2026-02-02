import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class EarlyFusionRanker(nn.Module):
    """
    Binary classifier that ranks candidate paragraphs based on their relevance to the question.
    Uses an 'Early Fusion' approach where Question and Paragraph are concatenated before processing.
    """

    def __init__(self, embedding_matrix=None):
        super(EarlyFusionRanker, self).__init__()

        # Initialize Embeddings
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=False,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(
                Config.VOCAB_SIZE, Config.EMBEDDING_DIM, padding_idx=0
            )

        # Convolutional Encoder Blocks
        # Block 1
        self.conv1 = nn.Conv1d(
            in_channels=Config.EMBEDDING_DIM,
            out_channels=Config.RANKER_FILTERS,
            kernel_size=Config.RANKER_KERNEL_SIZE,
            padding=Config.RANKER_KERNEL_SIZE // 2,
        )
        # Block 2
        self.conv2 = nn.Conv1d(
            in_channels=Config.RANKER_FILTERS,
            out_channels=Config.RANKER_FILTERS,
            kernel_size=Config.RANKER_KERNEL_SIZE,
            padding=Config.RANKER_KERNEL_SIZE // 2,
        )
        # Block 3
        self.conv3 = nn.Conv1d(
            in_channels=Config.RANKER_FILTERS,
            out_channels=Config.RANKER_FILTERS,
            kernel_size=Config.RANKER_KERNEL_SIZE,
            padding=Config.RANKER_KERNEL_SIZE // 2,
        )

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Classification Head
        self.fc1 = nn.Linear(Config.RANKER_FILTERS, Config.RANKER_HIDDEN_DIM)
        self.fc2 = nn.Linear(Config.RANKER_HIDDEN_DIM, 1)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (Batch, Seq_Len) - Concatenated [Q; SEP; Para]
        Returns:
            logits: (Batch,) - Unnormalized relevance scores
        """
        # Embedding: (Batch, Seq_Len, Emb_Dim)
        x = self.embedding(input_ids)

        # Transpose for Conv1d: (Batch, Emb_Dim, Seq_Len)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        # Convolutional Blocks
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        # We perform Global Max Pooling here to handle the remaining sequence length
        # (Batch, Filters, Reduced_Len) -> (Batch, Filters)
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)

        x = self.dropout(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        logits = self.fc2(x).squeeze(-1)
        return logits


class DynamicKernelReader(nn.Module):
    """
    Span extraction model that uses the Question to generate dynamic convolutional filters
    applied to the Paragraph.
    """

    def __init__(self, embedding_matrix=None):
        super(DynamicKernelReader, self).__init__()

        # Initialize Embeddings
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=False,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(
                Config.VOCAB_SIZE, Config.EMBEDDING_DIM, padding_idx=0
            )

        self.kernel_size = Config.READER_KERNEL_SIZE
        self.in_channels = Config.EMBEDDING_DIM
        self.out_channels = Config.READER_FILTERS

        # Total parameters needed for the dynamic kernel: Out * In * K
        self.num_kernel_params = self.out_channels * self.in_channels * self.kernel_size

        # Hypernetwork: Maps Question Vector -> Convolutional Weights
        self.hypernet = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM, Config.EMBEDDING_DIM * 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.EMBEDDING_DIM * 2, self.num_kernel_params),
        )

        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Output Heads for Start and End positions
        # 1x1 Conv maps feature channels to 1 logit per position
        self.start_head = nn.Conv1d(self.out_channels, 1, kernel_size=1)
        self.end_head = nn.Conv1d(self.out_channels, 1, kernel_size=1)

    def forward(self, q_input_ids, ctx_input_ids):
        """
        Args:
            q_input_ids: (Batch, Q_Len)
            ctx_input_ids: (Batch, Ctx_Len)
        Returns:
            start_logits: (Batch, Ctx_Len)
            end_logits: (Batch, Ctx_Len)
        """
        batch_size = q_input_ids.size(0)

        # --- 1. Question Encoding ---
        q_emb = self.embedding(q_input_ids)  # (B, QL, D)
        q_emb = self.dropout(q_emb)
        # Global Max Pool over question tokens to get a single vector
        q_vec = torch.max(q_emb, dim=1)[0]  # (B, D)

        # --- 2. Dynamic Kernel Generation ---
        # Generate weights from question vector
        raw_weights = self.hypernet(q_vec)  # (B, Out*In*K)

        # Reshape weights for grouped convolution
        # Shape: (Batch * Out_Channels, In_Channels, Kernel_Size)
        # This aligns with PyTorch's expectation for groups=Batch_Size
        weights = raw_weights.view(
            batch_size * self.out_channels, self.in_channels, self.kernel_size
        )

        # --- 3. Context Processing ---
        ctx_emb = self.embedding(ctx_input_ids)  # (B, CL, D)
        ctx_emb = self.dropout(ctx_emb)
        # Transpose for Conv1d: (B, D, CL)
        ctx_emb = ctx_emb.transpose(1, 2)

        # --- 4. Dynamic Convolution ---
        # We use the "groups" argument in Conv1d to apply a different kernel to each sample in the batch.
        # To do this, we reshape the input to treat the batch dimension as part of the channels.
        # Input shape becomes: (1, Batch * In_Channels, Ctx_Len)
        x = ctx_emb.reshape(1, batch_size * self.in_channels, -1)

        # Apply convolution
        # Groups = batch_size ensures that the i-th block of input channels (the i-th sample)
        # is convolved with the i-th block of output filters (generated from the i-th question).
        padding = self.kernel_size // 2
        features = F.conv1d(input=x, weight=weights, padding=padding, groups=batch_size)
        # Output shape: (1, Batch * Out_Channels, Ctx_Len)

        # Reshape back to (Batch, Out_Channels, Ctx_Len)
        features = features.view(batch_size, self.out_channels, -1)
        features = F.relu(features)
        features = self.dropout(features)

        # --- 5. Prediction ---
        start_logits = self.start_head(features).squeeze(1)  # (B, CL)
        end_logits = self.end_head(features).squeeze(1)  # (B, CL)

        return start_logits, end_logits
