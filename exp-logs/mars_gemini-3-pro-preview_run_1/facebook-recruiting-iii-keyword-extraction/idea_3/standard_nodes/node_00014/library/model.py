import torch
import torch.nn as nn
import torch.nn.functional as F

# Import torch_scatter for efficient segmented operations
# This package is available in the environment as per the task description.
try:
    from torch_scatter import scatter_softmax, scatter_sum
except ImportError:
    raise ImportError(
        "torch_scatter is required for Attention-DAN but is not installed."
    )


def offsets_to_batch_indices(offsets, total_tokens):
    """
    Converts a tensor of offsets (start positions) into a tensor of batch indices
    for every token in the flattened sequence.

    Example:
        offsets: [0, 2, 5] (Batch size 3, lengths 2, 3, ...)
        total_tokens: 8
        Returns: [0, 0, 1, 1, 1, 2, 2, 2]

    Args:
        offsets (torch.Tensor): [Batch_Size] Start indices of each sequence.
        total_tokens (int): Total number of tokens in the batch.

    Returns:
        torch.Tensor: [Total_Tokens] Batch index for each token.
    """
    device = offsets.device
    batch_size = offsets.size(0)

    if batch_size == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Calculate lengths of each sequence
    # We append the total token count to the end to compute the length of the last sequence
    # offsets are start indices, so length[i] = offsets[i+1] - offsets[i]
    end_points = torch.cat([offsets[1:], torch.tensor([total_tokens], device=device)])
    lengths = end_points - offsets

    # Generate batch indices: repeat the batch ID 'length' times
    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), lengths
    )

    return batch_indices


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Lightweight MLP for attention scores
        # Structure: Input -> W1 -> Tanh -> W2 -> Softmax
        # We project to the same dimension for the hidden attention state
        self.w1 = nn.Linear(embed_dim, embed_dim)
        self.w2 = nn.Linear(embed_dim, 1)

    def forward(self, embeddings, batch_indices, batch_size):
        """
        Computes attention-weighted average of embeddings.

        Args:
            embeddings: (Total_Tokens, Embed_Dim)
            batch_indices: (Total_Tokens) indicating which batch item each token belongs to.
            batch_size: int

        Returns:
            pooled: (Batch_Size, Embed_Dim)
        """
        # 1. Compute unnormalized attention scores
        # alpha = W2(tanh(W1(e)))
        # Shape: (Total_Tokens, Embed_Dim) -> (Total_Tokens, Embed_Dim) -> (Total_Tokens, 1)
        attn_logits = self.w2(torch.tanh(self.w1(embeddings))).squeeze(-1)

        # 2. Normalize scores using Softmax over each segment (sample)
        # torch_scatter.scatter_softmax handles the segmentation based on batch_indices
        # This computes softmax(x_i) where i belongs to the same group
        attn_weights = scatter_softmax(attn_logits, batch_indices, dim=0)

        # 3. Weighted Sum
        # Broadcast weights: (Total_Tokens, 1) * (Total_Tokens, Embed_Dim)
        weighted_embeddings = embeddings * attn_weights.unsqueeze(-1)

        # Sum embeddings belonging to the same batch index
        # dim_size ensures that if a batch item has 0 tokens, it returns a zero vector
        pooled = scatter_sum(
            weighted_embeddings, batch_indices, dim=0, dim_size=batch_size
        )

        return pooled


def initialize_weights(model, init_range=0.5):
    """
    Custom initialization strategy:
    - Backbone (Embeddings, Attention, FC1): Uniform [-0.5, 0.5] to maintain variance.
    - Head (FC2): Xavier Uniform for stable classification logits.
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            if "fc2" in name or "classifier" in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -init_range, init_range)
        elif "bias" in name:
            nn.init.zeros_(param)


class DualStreamAttentionDAN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=128,
        hidden_dim=1024,
        dropout=0.2,
        init_range=0.5,
        padding_idx=0,
    ):
        super().__init__()

        # 1. Shared Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # 2. Attention Pooling Modules for each stream (Title and Body)
        self.title_attn = AttentionPooling(embed_dim)
        self.body_attn = AttentionPooling(embed_dim)

        # 3. Wide Dense Layers
        # Input is concatenated Title + Body vectors (Embed_Dim * 2)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # 4. Initialization
        self.init_range = init_range
        initialize_weights(self, init_range)

    def forward(self, title_text, title_offsets, body_text, body_offsets):
        """
        Args:
            title_text: (Total_Title_Tokens) Flattened tensor of title token indices
            title_offsets: (Batch_Size) Start indices for titles
            body_text: (Total_Body_Tokens) Flattened tensor of body token indices
            body_offsets: (Batch_Size) Start indices for bodies

        Returns:
            logits: (Batch_Size, Num_Classes)
        """
        batch_size = title_offsets.size(0)

        # --- Process Title Stream ---
        # 1. Embed
        title_embeds = self.embedding(title_text)  # (Total_Title_Tokens, D)

        # 2. Generate Batch Indices for scatter operations
        title_batch_indices = offsets_to_batch_indices(
            title_offsets, title_text.size(0)
        )

        # 3. Attention Pooling
        title_vec = self.title_attn(
            title_embeds, title_batch_indices, batch_size
        )  # (B, D)

        # --- Process Body Stream ---
        # 1. Embed
        body_embeds = self.embedding(body_text)  # (Total_Body_Tokens, D)

        # 2. Generate Batch Indices
        body_batch_indices = offsets_to_batch_indices(body_offsets, body_text.size(0))

        # 3. Attention Pooling
        body_vec = self.body_attn(body_embeds, body_batch_indices, batch_size)  # (B, D)

        # --- Combine and Classify ---
        # Concatenate the two representations
        combined = torch.cat([title_vec, body_vec], dim=1)  # (B, 2*D)

        # Dense Layers
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
