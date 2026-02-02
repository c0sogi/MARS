import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from library.config import Config


class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder.
    Encodes the input character sequence into context vectors.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=Config.PAD_IDX
        )
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))
        # embedded: (batch_size, src_len, embedding_dim)

        outputs, hidden = self.gru(embedded)
        # outputs: (batch_size, src_len, hidden_dim * 2)
        # hidden: (n_layers * 2, batch_size, hidden_dim)

        return outputs, hidden


class AuxiliaryHead(nn.Module):
    """
    Auxiliary Classification Head.
    Predicts the token class (e.g., DATE, MONEY) from the Encoder's final state.
    """

    def __init__(self, hidden_dim, num_classes, dropout):
        super(AuxiliaryHead, self).__init__()
        # Input is concatenation of forward and backward final hidden states
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, encoder_hidden):
        # encoder_hidden: (n_layers * 2, batch_size, hidden_dim)

        # Take the last layer's forward and backward states
        # hidden[-2] is the last layer forward
        # hidden[-1] is the last layer backward
        hidden_forward = encoder_hidden[-2, :, :]
        hidden_backward = encoder_hidden[-1, :, :]

        # Concatenate: (batch_size, hidden_dim * 2)
        cat_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        x = self.fc1(cat_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        prediction = self.fc2(x)
        # prediction: (batch_size, num_classes)

        return prediction


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention.
    Calculates alignment scores between decoder hidden state and encoder outputs.
    """

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        # The decoder hidden state will be (batch, hidden_dim * 2) to match encoder output
        self.Wa = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # For decoder hidden
        self.Ua = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # For encoder outputs
        self.Va = nn.Linear(hidden_dim * 2, 1)

    def forward(self, query, keys, mask=None):
        # query: Decoder hidden state (batch_size, 1, dec_hidden_dim)
        # keys: Encoder outputs (batch_size, src_len, enc_hidden_dim * 2)
        # Note: dec_hidden_dim == enc_hidden_dim * 2 in this architecture

        # Calculate energy scores
        # query: (batch, 1, hidden*2) -> (batch, 1, hidden*2)
        # keys: (batch, seq, hidden*2) -> (batch, seq, hidden*2)

        # We broadcast query across the sequence length
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # scores: (batch_size, src_len, 1)

        scores = scores.squeeze(2)  # (batch_size, src_len)

        # Masking (optional but recommended for padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)
        # weights: (batch_size, src_len)

        # Calculate context vector
        # weights.unsqueeze(1): (batch, 1, src_len)
        # keys: (batch, src_len, hidden*2)
        # bmm -> (batch, 1, hidden*2)
        context = torch.bmm(weights.unsqueeze(1), keys)

        return context, weights


class Decoder(nn.Module):
    """
    Unidirectional GRU Decoder with Attention.
    Generates the normalized text sequence.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=Config.PAD_IDX
        )
        self.attention = BahdanauAttention(hidden_dim)

        # GRU input: Embedding + Context Vector
        # Context vector size is encoder output size (hidden_dim * 2)
        self.gru = nn.GRU(
            embedding_dim + (hidden_dim * 2),
            hidden_dim * 2,  # Decoder hidden size matches encoder output size
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, encoder_outputs, mask=None):
        # input_step: (batch_size, 1)
        # hidden: (n_layers, batch_size, hidden_dim * 2)
        # encoder_outputs: (batch_size, src_len, hidden_dim * 2)

        embedded = self.dropout(self.embedding(input_step))
        # embedded: (batch_size, 1, embedding_dim)

        # Calculate attention
        # Use the last layer of hidden state for attention query
        query = hidden[-1].unsqueeze(1)  # (batch, 1, hidden*2)
        context, attn_weights = self.attention(query, encoder_outputs, mask)
        # context: (batch_size, 1, hidden_dim * 2)

        # Concatenate embedding and context
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: (batch_size, 1, embedding_dim + hidden*2)

        output, hidden = self.gru(rnn_input, hidden)
        # output: (batch_size, 1, hidden_dim * 2)
        # hidden: (n_layers, batch_size, hidden_dim * 2)

        prediction = self.fc_out(output.squeeze(1))
        # prediction: (batch_size, vocab_size)

        return prediction, hidden, attn_weights


class MultiTaskSeq2Seq(nn.Module):
    """
    Main Model Class: Multi-Task Neuro-Symbolic Cascade (Neural Component).
    Combines Encoder, Decoder, and Auxiliary Head.
    """

    def __init__(self, vocab_size):
        super(MultiTaskSeq2Seq, self).__init__()

        self.vocab_size = vocab_size

        # Initialize components based on Config
        self.encoder = Encoder(
            vocab_size,
            Config.EMBEDDING_DIM,
            Config.HIDDEN_DIM,
            Config.ENC_LAYERS,
            Config.DROPOUT,
        )

        self.aux_head = AuxiliaryHead(
            Config.HIDDEN_DIM, Config.NUM_AUX_CLASSES, Config.DROPOUT
        )

        self.decoder = Decoder(
            vocab_size,
            Config.EMBEDDING_DIM,
            Config.HIDDEN_DIM,
            Config.DEC_LAYERS,
            Config.DROPOUT,
        )

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for both generation and classification.

        Args:
            src (torch.Tensor): Input sequences (batch_size, src_len).
            tgt (torch.Tensor, optional): Target sequences for teacher forcing (batch_size, tgt_len).
            teacher_forcing_ratio (float): Probability of using true target as next input.

        Returns:
            decoder_outputs (torch.Tensor): Generation logits (batch_size, max_len, vocab_size).
            aux_outputs (torch.Tensor): Classification logits (batch_size, num_classes).
        """
        batch_size = src.shape[0]
        max_len = Config.MAX_OUTPUT_LEN
        if tgt is not None:
            max_len = tgt.shape[1]

        # 1. Encode
        encoder_outputs, encoder_hidden = self.encoder(src)

        # 2. Auxiliary Task (Classification)
        aux_outputs = self.aux_head(encoder_hidden)

        # 3. Decode
        # Prepare decoder initial state
        # Encoder hidden is (n_layers*2, batch, hidden).
        # Decoder needs (n_layers, batch, hidden*2).
        # We concat the last forward and backward layers of encoder to form initial decoder state.
        # Assuming DEC_LAYERS = 1 for simplicity as per Config, or we project.
        # Here we take the last layer's fwd and bwd.

        hidden_forward = encoder_hidden[-2]
        hidden_backward = encoder_hidden[-1]
        decoder_hidden = torch.cat(
            (hidden_forward, hidden_backward), dim=1
        )  # (batch, hidden*2)
        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch, hidden*2)

        # If Decoder has more layers than 1, we might need to replicate,
        # but Config.DEC_LAYERS is 1.

        # Prepare first input (SOS Token)
        decoder_input = torch.tensor([[Config.SOS_IDX]], device=Config.DEVICE).repeat(
            batch_size, 1
        )

        # Create mask for attention (pad tokens in src should be ignored)
        # src shape: (batch, src_len)
        mask = src != Config.PAD_IDX

        # Store outputs
        decoder_outputs = torch.zeros(
            batch_size, max_len, self.vocab_size, device=Config.DEVICE
        )

        for t in range(max_len):
            output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )

            decoder_outputs[:, t, :] = output

            # Determine next input
            top1 = output.argmax(1)

            if tgt is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: use actual next token
                # tgt includes SOS, so tgt[:, t+1] is the next token?
                # Usually tgt is [SOS, c1, c2, EOS].
                # If t=0, we input SOS, output prediction for c1. Next input should be c1 (tgt[:, 1]).
                if t + 1 < max_len:
                    decoder_input = tgt[:, t + 1].unsqueeze(1)
                else:
                    # End of target sequence
                    break
            else:
                # Use predicted token
                decoder_input = top1.unsqueeze(1)

        return decoder_outputs, aux_outputs
