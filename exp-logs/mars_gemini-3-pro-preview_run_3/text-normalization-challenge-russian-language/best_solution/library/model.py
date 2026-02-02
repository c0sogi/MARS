import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from library.config import Config


class Encoder(nn.Module):
    """
    LSTM Encoder.
    Encodes the input sequence into a context vector and sequence of outputs.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        # embedded: [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        return outputs, hidden, cell


class Attention(nn.Module):
    """
    Bahdanau Attention Mechanism.
    Cite solution_lesson_node_00001: Addressing information bottleneck.
    """

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn_enc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.attn_dec = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim]

        # Calculate energy
        # [batch_size, src_len, dec_hid_dim]
        enc_energy = self.attn_enc(encoder_outputs)

        # [batch_size, 1, dec_hid_dim]
        dec_energy = self.attn_dec(hidden).unsqueeze(1)

        # [batch_size, src_len, dec_hid_dim]
        # Broadcasting addition
        energy = torch.tanh(enc_energy + dec_energy)

        # Calculate attention weights
        # [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    LSTM Decoder with Attention.
    Cite solution_lesson_node_00001: Using attention to query encoder outputs.
    """

    def __init__(
        self,
        output_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        n_layers,
        dropout,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # Input to LSTM is concatenation of embedding and weighted context
        self.rnn = nn.LSTM(
            enc_hid_dim + emb_dim,
            dec_hid_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, dec_hid_dim]
        # cell: [n_layers, batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim]

        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        # Calculate attention weights using the last layer's hidden state
        # hidden[-1]: [batch_size, dec_hid_dim]
        a = self.attention(hidden[-1], encoder_outputs)

        # a: [batch_size, 1, src_len]
        a = a.unsqueeze(1)

        # Calculate weighted context vector
        # [batch_size, 1, enc_hid_dim]
        weighted = torch.bmm(a, encoder_outputs)

        # Concatenate embedded input and weighted context
        # [batch_size, 1, emb_dim + enc_hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model wrapping Encoder and Decoder with Attention.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input is SOS
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            # Pass encoder_outputs to decoder
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

    def predict(self, src, sos_idx, eos_idx, max_len=100):
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]

            # Encode
            encoder_outputs, hidden, cell = self.encoder(src)

            input = torch.tensor(
                [sos_idx] * batch_size, dtype=torch.long, device=self.device
            )
            all_preds = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=self.device
            )
            all_preds[:, 0] = sos_idx
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for t in range(1, max_len):
                # Pass encoder_outputs
                output, hidden, cell = self.decoder(
                    input, hidden, cell, encoder_outputs
                )
                top1 = output.argmax(1)
                all_preds[:, t] = top1

                is_eos = top1 == eos_idx
                finished = finished | is_eos
                input = top1

                if finished.all():
                    all_preds = all_preds[:, : t + 1]
                    break

            return all_preds
