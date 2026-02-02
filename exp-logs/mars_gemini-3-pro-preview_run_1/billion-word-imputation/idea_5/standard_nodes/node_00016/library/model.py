import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import get_logger

logger = get_logger("model")


class GlobalLocalTransformer(nn.Module):
    """
    Transformer model designed for the Global-Localization task.
    Processes an interleaved sequence of words and [GAP] tokens.

    Outputs:
    1. Localization Logits: Probability that a specific [GAP] is the missing word location.
    2. Identification Logits: Probability distribution over the vocabulary for the missing word.
    3. Hidden States: Raw representations for Latent Alignment regularization.
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from Config
        self.vocab_size = Config.VOCAB_SIZE
        self.d_model = Config.EMBED_DIM
        self.nhead = Config.NUM_HEADS
        self.num_layers = Config.NUM_LAYERS
        self.dropout = Config.DROPOUT
        self.max_len = Config.MAX_LEN

        logger.info(
            f"Initializing GlobalLocalTransformer with d_model={self.d_model}, layers={self.num_layers}"
        )

        # 1. Embeddings
        # Padding index is 0 as per Vocabulary implementation
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.pos_encoder = nn.Embedding(self.max_len, self.d_model)
        self.emb_dropout = nn.Dropout(self.dropout)

        # 2. Backbone
        # Standard Transformer Encoder
        # dim_feedforward is set to 4 * d_model following standard practices
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # 3. Global Localization Head
        # Projects hidden state to a single scalar logit for "gap-ness"
        self.loc_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.Tanh(), nn.Linear(self.d_model, 1)
        )

        # 4. Identification Head
        # Projects hidden state to vocabulary distribution
        # bias=False is required for weight tying
        self.id_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # 5. Weight Tying
        # Tie the identification head weights to the input embedding weights.
        # This constrains the output space to be semantically close to the input space,
        # which aids the Latent Alignment objective.
        self.id_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with a fixed range for stability."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_encoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Localization Head
        self.loc_head[0].bias.data.zero_()
        self.loc_head[0].weight.data.uniform_(-initrange, initrange)
        self.loc_head[2].bias.data.zero_()
        self.loc_head[2].weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, padding_mask=None):
        """
        Args:
            input_ids (Tensor): Shape (Batch, Seq_Len). Indices of tokens.
            padding_mask (Tensor, optional): Shape (Batch, Seq_Len).
                                             True indicates padding (ignored positions).
                                             If None, generated from input_ids == 0.

        Returns:
            loc_logits (Tensor): Shape (Batch, Seq_Len). Logits for gap localization.
            id_logits (Tensor): Shape (Batch, Seq_Len, Vocab_Size). Logits for word identification.
            hidden_states (Tensor): Shape (Batch, Seq_Len, Embed_Dim). Raw hidden states.
        """
        batch_size, seq_len = input_ids.size()

        # Generate position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # 1. Embedding Layer
        # Scale embeddings by sqrt(d_model) as per Attention is All You Need
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoder(positions)
        x = self.emb_dropout(x)

        # 2. Transformer Encoder
        # Create padding mask if not provided (0 is PAD token)
        if padding_mask is None:
            padding_mask = input_ids == 0

        hidden_states = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # 3. Heads

        # Localization: (Batch, Seq_Len, Embed_Dim) -> (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
        loc_logits = self.loc_head(hidden_states).squeeze(-1)

        # Identification: (Batch, Seq_Len, Embed_Dim) -> (Batch, Seq_Len, Vocab_Size)
        id_logits = self.id_head(hidden_states)

        return loc_logits, id_logits, hidden_states

    def get_input_embeddings(self):
        """
        Returns the embedding layer.
        Used by the training loop to retrieve target word embeddings for Latent Alignment loss.
        """
        return self.embedding
