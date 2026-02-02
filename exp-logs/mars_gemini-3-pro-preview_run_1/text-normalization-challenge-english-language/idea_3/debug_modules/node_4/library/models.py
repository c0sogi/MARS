import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from library.config import Config


class CharCNN(nn.Module):
    """
    Character-level CNN for extracting morphological features from tokens.
    Input: (batch_size, seq_len, char_len)
    Output: (batch_size, seq_len, filters)
    """

    def __init__(self, vocab_size):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.CHAR_EMBEDDING_DIM,
            padding_idx=0,
        )
        self.conv = nn.Conv1d(
            in_channels=Config.CHAR_EMBEDDING_DIM,
            out_channels=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            padding=1,  # Padding to handle short tokens
        )
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

    def forward(self, char_ids):
        # char_ids: (batch, seq_len, char_len)
        batch_size, seq_len, char_len = char_ids.size()

        # Reshape to treat every token in the batch/sequence as an independent sample
        # (N, L) -> (batch * seq_len, char_len)
        x = char_ids.view(-1, char_len)

        # Embedding: (N, L, D)
        x = self.embedding(x)

        # Permute for CNN: (N, D, L)
        x = x.permute(0, 2, 1)

        # 1D Convolution
        x = self.conv(x)  # (N, filters, L_out)
        x = F.relu(x)

        # Global Max Pooling over character dimension
        x, _ = torch.max(x, dim=2)  # (N, filters)

        # Reshape back to sequence format: (batch, seq_len, filters)
        x = x.view(batch_size, seq_len, -1)

        return self.dropout(x)


class BiLSTMTagger(nn.Module):
    """
    Bidirectional LSTM Tagger for token classification.
    Combines Word Embeddings and CharCNN features.
    """

    def __init__(self, token_vocab_size, char_vocab_size, num_classes):
        super(BiLSTMTagger, self).__init__()

        # Word Embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=token_vocab_size,
            embedding_dim=Config.TAGGER_EMBEDDING_DIM,
            padding_idx=0,
        )

        # Character Feature Extractor
        self.char_cnn = CharCNN(char_vocab_size)

        # Calculate LSTM input dimension
        lstm_input_dim = Config.TAGGER_EMBEDDING_DIM + Config.CNN_FILTERS

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=Config.TAGGER_HIDDEN_DIM,
            num_layers=Config.TAGGER_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_LAYERS > 1 else 0,
        )

        # Output Projection
        # Hidden size is doubled because of bidirectionality
        self.fc = nn.Linear(Config.TAGGER_HIDDEN_DIM * 2, num_classes)
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

    def forward(self, word_ids, char_ids, lengths=None):
        """
        Args:
            word_ids: (batch, seq_len)
            char_ids: (batch, seq_len, char_len)
            lengths: (batch) - Actual lengths of sequences for packing
        """
        # 1. Get Features
        word_emb = self.word_embedding(word_ids)  # (batch, seq, word_dim)
        char_feat = self.char_cnn(char_ids)  # (batch, seq, char_dim)

        # 2. Concatenate
        combined = torch.cat([word_emb, char_feat], dim=2)  # (batch, seq, input_dim)
        combined = self.dropout(combined)

        # 3. Bi-LSTM Processing
        if lengths is not None:
            # Pack sequence for efficiency and ignoring padding
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(
                combined, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Fallback if lengths not provided (assumes full padding mask handling by caller if needed)
            output, _ = self.lstm(combined)

        # 4. Classification
        logits = self.fc(self.dropout(output))  # (batch, seq, num_classes)

        return logits


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(
            input_size, Config.SEQ2SEQ_EMBED_DIM, padding_idx=0
        )
        self.lstm = nn.LSTM(Config.SEQ2SEQ_EMBED_DIM, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # input: (batch, seq)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    """
    Dot-product attention (Bahdanau-style concatenation variant).
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch, hidden) -> needs to be (batch, seq, hidden) for broadcasting
        # encoder_outputs: (batch, seq, hidden)

        seq_len = encoder_outputs.size(1)

        # Repeat hidden state for each time step
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, seq)

        return F.softmax(attention, dim=1)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(
            output_size, Config.SEQ2SEQ_EMBED_DIM, padding_idx=0
        )
        self.attention = Attention(hidden_size)

        # Input to LSTM is embedding + context vector
        self.lstm = nn.LSTM(
            Config.SEQ2SEQ_EMBED_DIM + hidden_size, hidden_size, batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: (batch) -> single step
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input))  # (batch, 1, embed)

        # Calculate Attention
        attn_weights = self.attention(hidden, encoder_outputs)  # (batch, seq)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, seq)

        # Apply Attention
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden)

        # Combine embedding and context
        rnn_input = torch.cat((embedded, context), dim=2)

        # LSTM Step
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        # Prediction
        prediction = self.out(output.squeeze(1))  # (batch, output_size)

        return prediction, hidden, cell, attn_weights


class Seq2SeqNormalizer(nn.Module):
    """
    Character-level Seq2Seq model with Attention for text normalization.
    """

    def __init__(self, char_vocab_size):
        super(Seq2SeqNormalizer, self).__init__()
        self.vocab_size = char_vocab_size

        self.encoder = EncoderRNN(
            char_vocab_size, Config.SEQ2SEQ_HIDDEN_DIM, Config.SEQ2SEQ_DROPOUT
        )
        self.decoder = DecoderRNN(
            char_vocab_size, Config.SEQ2SEQ_HIDDEN_DIM, Config.SEQ2SEQ_DROPOUT
        )

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Forward pass for training.
        src: (batch, src_len)
        tgt: (batch, tgt_len) - Includes <sos> and <eos>
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.vocab_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)

        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input is <sos>
        input = tgt[:, 0]

        # Decode loop
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            # Teacher Forcing vs Autoregression
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

    def predict(self, src, sos_idx, eos_idx, max_len=None):
        """
        Inference method for generating normalized text.
        """
        self.eval()
        if max_len is None:
            max_len = Config.MAX_OUTPUT_LEN

        batch_size = src.size(0)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)

            # Start with <sos>
            input = torch.tensor([sos_idx] * batch_size, device=src.device)

            # Track finished sequences
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
            predictions = []

            for t in range(max_len):
                output, hidden, cell, _ = self.decoder(
                    input, hidden, cell, encoder_outputs
                )
                top1 = output.argmax(1)

                predictions.append(top1.unsqueeze(1))
                input = top1

                # Check EOS
                is_eos = top1 == eos_idx
                finished = finished | is_eos

                # If all sequences hit EOS, stop early
                if finished.all():
                    break

            if len(predictions) > 0:
                predictions = torch.cat(predictions, dim=1)
            else:
                predictions = torch.zeros(
                    batch_size, 0, device=src.device, dtype=torch.long
                )

        return predictions
