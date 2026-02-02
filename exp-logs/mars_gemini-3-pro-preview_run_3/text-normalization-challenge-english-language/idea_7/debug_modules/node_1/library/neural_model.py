import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from library.config import Config


class FactoredEmbedding(nn.Module):
    """
    Concatenates embeddings for Character ID, Case, and Type.
    """

    def __init__(self):
        super(FactoredEmbedding, self).__init__()

        self.char_emb = nn.Embedding(
            Config.VOCAB_SIZE, Config.CHAR_EMB_DIM, padding_idx=Config.PAD_IDX
        )
        self.case_emb = nn.Embedding(
            Config.CASE_VOCAB_SIZE, Config.CASE_EMB_DIM, padding_idx=0
        )
        self.type_emb = nn.Embedding(
            Config.TYPE_VOCAB_SIZE, Config.TYPE_EMB_DIM, padding_idx=0
        )

        self.output_dim = (
            Config.CHAR_EMB_DIM + Config.CASE_EMB_DIM + Config.TYPE_EMB_DIM
        )
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, char_idx, case_idx, type_idx):
        # char_idx: [batch, seq_len]
        e_char = self.char_emb(char_idx)
        e_case = self.case_emb(case_idx)
        e_type = self.type_emb(type_idx)

        # Concatenate along the embedding dimension
        combined = torch.cat([e_char, e_case, e_type], dim=2)
        return self.dropout(combined)


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention.
    """

    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super(Attention, self).__init__()

        # Encoder outputs are bidirectional, so dim is enc_hid_dim * 2
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch, dec_hid_dim] (current decoder hidden state)
        # encoder_outputs: [batch, src_len, enc_hid_dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        # hidden: [batch, src_len, dec_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate energy
        # energy: [batch, src_len, attn_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention scores
        # attention: [batch, src_len]
        attention = self.v(energy).squeeze(2)

        if mask is not None:
            # Mask is typically [batch, src_len], 1 for valid, 0 for pad
            # We want to mask out pads by setting attention to -inf
            attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = FactoredEmbedding()
        self.rnn = nn.GRU(
            input_size=self.embedding.output_dim,
            hidden_size=Config.ENC_HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
        )

        # Fully connected layer to project concatenated bidirectional hidden states
        # to the decoder hidden size for initialization
        self.fc_hidden = nn.Linear(Config.ENC_HIDDEN_DIM * 2, Config.DEC_HIDDEN_DIM)

        # Auxiliary Classifier Head
        # Input: Concatenated last forward and backward hidden states
        self.aux_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.ENC_HIDDEN_DIM * 2, Config.ENC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.ENC_HIDDEN_DIM, Config.NUM_CLASSES),
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, src_char, src_case, src_type):
        # Embed
        embedded = self.embedding(src_char, src_case, src_type)

        # RNN
        # outputs: [batch, src_len, enc_hid_dim * 2]
        # hidden: [num_layers * 2, batch, enc_hid_dim]
        outputs, hidden = self.rnn(embedded)

        # Prepare initial decoder hidden state
        # We take the hidden state from the last layer
        # hidden[-2, :, :] is the last forward layer
        # hidden[-1, :, :] is the last backward layer
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]

        # cat_hidden: [batch, enc_hid_dim * 2]
        cat_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Project to decoder hidden size
        dec_init_hidden = torch.tanh(self.fc_hidden(cat_hidden))

        # Auxiliary Classification
        aux_logits = self.aux_classifier(cat_hidden)

        return outputs, dec_init_hidden, aux_logits


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.attention = Attention(
            Config.ENC_HIDDEN_DIM, Config.DEC_HIDDEN_DIM, Config.ATTN_DIM
        )

        # Decoder input is just character ID (no factors for target generation)
        self.embedding = nn.Embedding(
            Config.VOCAB_SIZE, Config.CHAR_EMB_DIM, padding_idx=Config.PAD_IDX
        )

        # Input to GRU is embedding + context vector
        self.rnn = nn.GRU(
            input_size=Config.CHAR_EMB_DIM + (Config.ENC_HIDDEN_DIM * 2),
            hidden_size=Config.DEC_HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,  # Usually decoder matches encoder layers, but here we use 1 effective layer logic for state or match config
            bidirectional=False,
            batch_first=True,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
        )

        self.fc_out = nn.Linear(
            Config.CHAR_EMB_DIM + Config.DEC_HIDDEN_DIM + (Config.ENC_HIDDEN_DIM * 2),
            Config.VOCAB_SIZE,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, input_idx, hidden, encoder_outputs, mask):
        # input_idx: [batch] (one time step)
        # hidden: [num_layers, batch, dec_hid_dim] (current hidden state)
        # encoder_outputs: [batch, src_len, enc_hid_dim * 2]
        # mask: [batch, src_len]

        input_idx = input_idx.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_idx))  # [batch, 1, char_emb_dim]

        # Calculate attention weights
        # We use the top layer of hidden state for attention calculation if multi-layer
        # hidden shape is [num_layers, batch, dim]. We take hidden[-1]
        attn_weights = self.attention(
            hidden[-1], encoder_outputs, mask
        )  # [batch, src_len]

        # Calculate context vector
        # attn_weights: [batch, 1, src_len]
        attn_weights = attn_weights.unsqueeze(1)

        # context: [batch, 1, enc_hid_dim * 2]
        context = torch.bmm(attn_weights, encoder_outputs)

        # Combine embedding and context for RNN input
        rnn_input = torch.cat((embedded, context), dim=2)

        # Forward pass through GRU
        output, hidden = self.rnn(rnn_input, hidden)

        # output: [batch, 1, dec_hid_dim]
        # hidden: [num_layers, batch, dec_hid_dim]

        # Prediction
        # Concatenate embedding, output, and context to make prediction
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))

        return prediction, hidden, attn_weights.squeeze(1)


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = Config.DEVICE

    def forward(
        self, src_char, src_case, src_type, tgt=None, teacher_forcing_ratio=0.5
    ):
        # src_char: [batch, src_len]
        # tgt: [batch, tgt_len]

        batch_size = src_char.shape[0]
        max_len = Config.MAX_LEN if tgt is None else tgt.shape[1]

        # Create mask for attention (1 for non-pad, 0 for pad)
        # src_char contains PAD_IDX
        mask = (src_char != Config.PAD_IDX).long()

        # Encode
        encoder_outputs, hidden, aux_logits = self.encoder(src_char, src_case, src_type)

        # Prepare Decoder Initial Hidden State
        # The encoder returns a projected single layer hidden state: [batch, dec_hid_dim]
        # The decoder expects: [num_layers, batch, dec_hid_dim]
        # We replicate it for all layers
        hidden = hidden.unsqueeze(0).repeat(Config.NUM_LAYERS, 1, 1)

        # First input to decoder is SOS_TOKEN
        decoder_input = torch.tensor([Config.SOS_IDX] * batch_size, device=self.device)

        outputs = torch.zeros(
            batch_size, max_len, Config.VOCAB_SIZE, device=self.device
        )

        # Decoding Loop
        for t in range(1, max_len):  # Start from 1 because 0 is implicitly SOS
            output, hidden, _ = self.decoder(
                decoder_input, hidden, encoder_outputs, mask
            )

            outputs[:, t, :] = output

            # Decide next input
            top1 = output.argmax(1)

            if tgt is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: use actual next token
                decoder_input = tgt[:, t]
            else:
                # Use predicted token
                decoder_input = top1

        return outputs, aux_logits
