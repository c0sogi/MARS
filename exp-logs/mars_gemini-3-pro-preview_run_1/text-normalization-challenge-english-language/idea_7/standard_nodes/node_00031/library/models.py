import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CharCNN(nn.Module):
    """
    Extracts morphological features from character sequences using a 1D Convolutional Neural Network.
    Input: Character indices of shape (Batch * Seq_Len, Max_Word_Len)
    Output: Fixed-size feature vector of shape (Batch * Seq_Len, Filters)
    """

    def __init__(self, num_chars):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_chars, Config.TAGGER_CHAR_EMBEDDING_DIM, padding_idx=0
        )

        self.conv = nn.Conv1d(
            in_channels=Config.TAGGER_CHAR_EMBEDDING_DIM,
            out_channels=Config.TAGGER_CNN_FILTERS,
            kernel_size=Config.TAGGER_CNN_KERNEL_SIZE,
            padding=1,  # Padding to maintain approximate length for small kernels
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (N, L) where N = Batch * Seq_Len, L = Max_Word_Len

        # Embedding: (N, L, D_char)
        x = self.embedding(x)

        # Permute for Conv1d: (N, D_char, L)
        x = x.permute(0, 2, 1)

        # Conv1d: (N, Filters, L_out)
        x = self.conv(x)
        x = self.relu(x)

        # Global Max Pooling over the sequence length dimension
        # (N, Filters)
        x, _ = torch.max(x, dim=2)

        return x


class AttentionBiLSTMTagger(nn.Module):
    """
    Sequence Tagger combining Word Embeddings, CharCNN features, Bi-LSTM, and Multi-Head Attention.

    Structure:
    1. Inputs: Word Indices, Character Indices
    2. Features: Concat(Word_Emb, CharCNN(Char_Indices))
    3. Context: Bi-LSTM
    4. Global Context: Multi-Head Self-Attention (on LSTM outputs)
    5. Output: Linear Classifier -> Class Logits
    """

    def __init__(self, num_tokens, num_chars, num_classes):
        super(AttentionBiLSTMTagger, self).__init__()

        # 1. Feature Extraction
        self.word_embedding = nn.Embedding(
            num_tokens, Config.TAGGER_EMBEDDING_DIM, padding_idx=0
        )
        self.char_cnn = CharCNN(num_chars)

        # Input dimension for LSTM
        input_dim = Config.TAGGER_EMBEDDING_DIM + Config.TAGGER_CNN_FILTERS

        # 2. Recurrent Layer (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.TAGGER_HIDDEN_DIM,
            num_layers=Config.TAGGER_RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.TAGGER_USE_BIDIRECTIONAL,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_RNN_LAYERS > 1 else 0,
        )

        lstm_output_dim = Config.TAGGER_HIDDEN_DIM * (
            2 if Config.TAGGER_USE_BIDIRECTIONAL else 1
        )

        # 3. Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=Config.TAGGER_ATTENTION_HEADS,
            dropout=Config.TAGGER_DROPOUT,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # 4. Classification Head
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, token_ids, char_ids, mask=None):
        """
        Args:
            token_ids: (Batch, Seq_Len)
            char_ids: (Batch, Seq_Len, Max_Word_Len)
            mask: (Batch, Seq_Len) - Optional mask where 1 is valid, 0 is pad.
        """
        batch_size, seq_len = token_ids.size()

        # Word Embeddings: (B, S, D_word)
        word_emb = self.word_embedding(token_ids)

        # Char Features
        # Flatten batch and sequence dims for CharCNN: (B*S, W)
        char_ids_flat = char_ids.view(-1, char_ids.size(-1))
        char_feats_flat = self.char_cnn(char_ids_flat)
        # Reshape back: (B, S, D_cnn)
        char_feats = char_feats_flat.view(batch_size, seq_len, -1)

        # Concatenate features: (B, S, D_word + D_cnn)
        combined_input = torch.cat([word_emb, char_feats], dim=-1)

        # LSTM Pass
        # lstm_out: (B, S, Hidden*2)
        lstm_out, _ = self.lstm(combined_input)

        # Attention Pass
        # Prepare key_padding_mask for MultiheadAttention (True for pad, False for valid)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0

        # Self-Attention (Query=Key=Value=LSTM_Out)
        attn_out, _ = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            key_padding_mask=key_padding_mask,
        )

        # Residual Connection + LayerNorm
        # (B, S, Hidden*2)
        context_vectors = self.layer_norm(lstm_out + attn_out)

        # Classification
        x = self.dropout(context_vectors)
        logits = self.classifier(x)  # (B, S, Num_Classes)

        return logits


class TransformerSeq2Seq(nn.Module):
    """
    Transformer Encoder-Decoder for character-level text normalization.
    Conditioned on the token class to guide generation.
    """

    def __init__(self, num_chars, num_classes):
        super(TransformerSeq2Seq, self).__init__()

        self.d_model = Config.SEQ2SEQ_EMBEDDING_DIM

        # Embeddings
        self.char_embedding = nn.Embedding(num_chars, self.d_model, padding_idx=0)
        self.class_embedding = nn.Embedding(num_classes, self.d_model)

        self.pos_encoder = PositionalEncoding(
            self.d_model, dropout=Config.SEQ2SEQ_DROPOUT
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=Config.SEQ2SEQ_HEADS,
            num_encoder_layers=Config.SEQ2SEQ_LAYERS,
            num_decoder_layers=Config.SEQ2SEQ_LAYERS,
            dim_feedforward=Config.SEQ2SEQ_HIDDEN_DIM,
            dropout=Config.SEQ2SEQ_DROPOUT,
            batch_first=True,
        )

        # Output Head
        self.fc_out = nn.Linear(self.d_model, num_chars)

    def forward(
        self, src, tgt, class_ids, src_key_padding_mask=None, tgt_key_padding_mask=None
    ):
        """
        Args:
            src: (Batch, Src_Len) - Source character sequence
            tgt: (Batch, Tgt_Len) - Target character sequence (input to decoder)
            class_ids: (Batch,) - Class ID for each sample
            src_key_padding_mask: (Batch, Src_Len) - True where src is padded
            tgt_key_padding_mask: (Batch, Tgt_Len) - True where tgt is padded
        """
        # 1. Source Embedding with Class Conditioning
        # (B, S_src, D)
        src_emb = self.char_embedding(src) * math.sqrt(self.d_model)

        # Retrieve Class Embedding: (B, D)
        class_emb = self.class_embedding(class_ids)

        # Add Class Embedding to Source Embeddings
        # Unsqueeze to broadcast: (B, 1, D) + (B, S_src, D) -> (B, S_src, D)
        # This conditions the entire encoder sequence on the class
        src_emb = src_emb + class_emb.unsqueeze(1)

        # Add Positional Encoding
        src_emb = self.pos_encoder(src_emb)

        # 2. Target Embedding
        # (B, S_tgt, D)
        tgt_emb = self.char_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 3. Generate Masks
        # Causal mask for the decoder to prevent attending to future tokens
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            src.device
        )

        # 4. Transformer Forward Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # 5. Project to Vocabulary
        logits = self.fc_out(output)

        return logits
