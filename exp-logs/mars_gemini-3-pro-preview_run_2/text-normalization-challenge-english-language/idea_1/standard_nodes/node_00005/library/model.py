import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from library.config import Config


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    Encodes the input sequence into context vectors.
    """

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len=None):
        # src: [batch_size, src_len]

        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]

        # If src_len is provided, we could use pack_padded_sequence here for efficiency
        # For this baseline, we proceed with standard processing

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, enc_hid_dim * 2]
        # hidden: [n_layers * n_directions, batch_size, enc_hid_dim]

        # Prepare hidden state for the decoder (which is unidirectional)
        # We concatenate the forward and backward hidden states of the last layer
        # hidden[-2, :, :] is the last of the forwards RNN
        # hidden[-1, :, :] is the last of the backwards RNN

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )
        # hidden: [batch_size, dec_hid_dim]

        return outputs, hidden


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention.
    Calculates alignment scores between decoder hidden state and encoder outputs.
    """

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, dec_hid_dim]

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, dec_hid_dim]

        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]

        # Apply mask if provided (mask out padding tokens)
        if mask is not None:
            # mask: [batch_size, src_len] (1 for valid, 0 for pad)
            # We want to set attention to -inf where mask is 0
            attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    Unidirectional LSTM Decoder with Attention.
    Generates the output sequence character by character.
    """

    def __init__(
        self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention
    ):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: [batch_size] (one character index per batch)
        # hidden: [batch_size, dec_hid_dim] (previous hidden state)
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        # mask: [batch_size, src_len]

        input = input.unsqueeze(1)
        # input: [batch_size, 1]

        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]

        # Calculate attention weights
        a = self.attention(hidden, encoder_outputs, mask)
        # a: [batch_size, src_len]

        a = a.unsqueeze(1)
        # a: [batch_size, 1, src_len]

        # Apply attention to encoder outputs to get weighted sum (context vector)
        weighted = torch.bmm(a, encoder_outputs)
        # weighted: [batch_size, 1, enc_hid_dim * 2]

        # Concatenate weighted context and embedding for LSTM input
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [batch_size, 1, (enc_hid_dim * 2) + emb_dim]

        # Pass through LSTM
        # We need to unsqueeze hidden to match LSTM expectation: [num_layers, batch, hidden]
        output, (hidden, cell) = self.rnn(
            rnn_input, (hidden.unsqueeze(0), torch.zeros_like(hidden.unsqueeze(0)))
        )

        # output: [batch_size, 1, dec_hid_dim]
        # hidden: [1, batch_size, dec_hid_dim] -> squeeze back to [batch_size, dec_hid_dim]
        hidden = hidden.squeeze(0)

        # Prediction
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence Model Wrapper.
    Encapsulates Encoder and Decoder and handles the forward pass logic.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # src_len: [batch_size]
        # tgt: [batch_size, tgt_len]
        # teacher_forcing_ratio: float

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # Encode
        encoder_outputs, hidden = self.encoder(src, src_len)

        # Create mask for attention (1 for valid tokens, 0 for pad)
        # src is [batch, src_len], Config.PAD_IDX is 0
        mask = src != Config.PAD_IDX

        # First input to decoder is the <sos> token
        input = tgt[:, 0]

        # Loop through target sequence
        # Start from index 1 because index 0 is <sos>
        for t in range(1, tgt_len):

            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)

            outputs[:, t] = output

            # Decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = tgt[:, t] if teacher_force else top1

        return outputs

    def predict(self, src, src_len, max_len=None):
        """
        Inference method for generating predictions without ground truth.
        Uses greedy decoding.
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            if max_len is None:
                max_len = Config.MAX_SEQ_LEN

            vocab_size = self.decoder.output_dim

            # Encode
            encoder_outputs, hidden = self.encoder(src, src_len)
            mask = src != Config.PAD_IDX

            # Start token
            input = torch.tensor([Config.SOS_IDX] * batch_size, device=self.device)

            # Store predictions
            predictions = []

            for t in range(max_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs, mask)

                # Greedy selection
                top1 = output.argmax(1)

                predictions.append(top1.unsqueeze(1))

                input = top1

            # Concatenate predictions: [batch_size, max_len]
            return torch.cat(predictions, dim=1)
