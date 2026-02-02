import torch
import torch.nn as nn
import random
from library.config import Config


class Encoder(nn.Module):
    """
    Vanilla LSTM Encoder.
    Encodes the input sequence into a context vector (hidden and cell states).
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

        # Return hidden and cell states to be used as initial context for decoder
        return hidden, cell


class Decoder(nn.Module):
    """
    Vanilla LSTM Decoder.
    Decodes the context vector into a sequence of characters.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size] (indices of current char)
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]

        # input: [batch_size, 1]
        input = input.unsqueeze(1)

        # embedded: [batch_size, 1, emb_dim]
        embedded = self.dropout(self.embedding(input))

        # output: [batch_size, 1, hid_dim]
        # hidden, cell: [n_layers, batch_size, hid_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction: [batch_size, output_dim]
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model wrapping Encoder and Decoder.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Forward pass for training with Teacher Forcing.

        Args:
            src (torch.Tensor): Source sequence [batch_size, src_len]
            tgt (torch.Tensor): Target sequence [batch_size, tgt_len]
            teacher_forcing_ratio (float): Probability to use teacher forcing.

        Returns:
            outputs (torch.Tensor): [batch_size, tgt_len, output_dim]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encode source sequence
        hidden, cell = self.encoder(src)

        # First input to the decoder is the <SOS> token
        input = tgt[:, 0]

        # Loop through target sequence (starting from index 1, as index 0 is SOS)
        # However, we usually store the prediction for position t at index t.
        # The output at t=0 corresponds to predicting the token after SOS.
        # But to match shapes, we often just fill from t=1 or handle SOS differently.
        # Standard practice:
        # Input at t=0 is SOS. Output is prediction for t=1.
        # We store this prediction at outputs[:, 1, :].
        # outputs[:, 0, :] remains 0 (or can be set to SOS probability 1.0 if needed, usually ignored by loss).

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Store prediction
            outputs[:, t, :] = output

            # Decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = tgt[:, t] if teacher_force else top1

        return outputs

    def predict(self, src, sos_idx, eos_idx, max_len=100):
        """
        Inference method using greedy decoding.

        Args:
            src (torch.Tensor): Source sequence [batch_size, src_len]
            sos_idx (int): Index of Start of Sequence token.
            eos_idx (int): Index of End of Sequence token.
            max_len (int): Maximum length of generated sequence.

        Returns:
            list[list[int]]: Predicted indices for each sample in batch.
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]

            # Encode
            hidden, cell = self.encoder(src)

            # Initialize input with SOS
            input = torch.tensor(
                [sos_idx] * batch_size, dtype=torch.long, device=self.device
            )

            # Store predictions
            # List of lists to handle variable lengths if we were stopping early per sequence,
            # but for batch efficiency we usually generate fixed tensor and cut later.
            # Here we will return the tensor indices.

            all_preds = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=self.device
            )
            # Set first column to SOS (optional, but good for consistency)
            all_preds[:, 0] = sos_idx

            # Track which sequences have finished
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for t in range(1, max_len):
                output, hidden, cell = self.decoder(input, hidden, cell)

                # Greedy selection
                top1 = output.argmax(1)

                # Store prediction
                all_preds[:, t] = top1

                # Update finished status
                is_eos = top1 == eos_idx
                finished = finished | is_eos

                # Next input
                input = top1

                # If all sequences finished, stop early
                if finished.all():
                    # Trim the tensor to current length
                    all_preds = all_preds[:, : t + 1]
                    break

            return all_preds
