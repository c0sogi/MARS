import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from library.config import Config


class CharEncoder(nn.Module):
    """
    Encodes character sequences using 1D-CNN to extract morphological features.
    Input: (Batch, Seq_Len, Char_Len)
    Output: (Batch, Seq_Len, Filters)
    """

    def __init__(self, num_chars, embed_dim, filters, kernel_size, padding_idx=0):
        super(CharEncoder, self).__init__()
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=padding_idx)

        # 1D Convolution to capture n-gram character patterns
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Maintain sequence length roughly
        )
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

    def forward(self, char_ids):
        # char_ids: (Batch, Seq_Len, Char_Len)
        batch_size, seq_len, char_len = char_ids.size()

        # Flatten to process all tokens in the batch/sequence in parallel
        # Shape: (Batch * Seq_Len, Char_Len)
        flat_inputs = char_ids.view(-1, char_len)

        # Embed: (Batch * Seq_Len, Char_Len, Embed_Dim)
        embedded = self.embedding(flat_inputs)

        # Permute for CNN: (Batch * Seq_Len, Embed_Dim, Char_Len)
        embedded = embedded.permute(0, 2, 1)

        # Conv1D
        conved = self.conv1d(embedded)  # (Batch * Seq_Len, Filters, L_out)
        conved = F.relu(conved)

        # Global Max Pooling over character sequence to get fixed representation
        # Kernel size equals the length of the convolution output
        pooled = F.max_pool1d(conved, kernel_size=conved.shape[2]).squeeze(
            2
        )  # (Batch * Seq_Len, Filters)

        pooled = self.dropout(pooled)

        # Reshape back to original sequence structure
        # (Batch, Seq_Len, Filters)
        return pooled.view(batch_size, seq_len, -1)


class BiLSTMTagger(nn.Module):
    """
    Bi-LSTM Tagger that combines Word Embeddings and Character-level CNN features.
    """

    def __init__(self, vocab_size, num_classes, char_vocab_size):
        super(BiLSTMTagger, self).__init__()

        # Word Embedding
        self.word_embedding = nn.Embedding(
            vocab_size, Config.TAGGER_EMBED_DIM, padding_idx=0
        )

        # Character Encoder
        self.char_encoder = CharEncoder(
            num_chars=char_vocab_size,
            embed_dim=Config.TAGGER_CHAR_EMBED_DIM,
            filters=Config.TAGGER_CHAR_CNN_FILTERS,
            kernel_size=Config.TAGGER_CHAR_CNN_KERNEL_SIZE,
            padding_idx=0,
        )

        # Calculate input dimension for LSTM (Word Embed + Char Features)
        lstm_input_dim = Config.TAGGER_EMBED_DIM + Config.TAGGER_CHAR_CNN_FILTERS

        # Bi-LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=Config.TAGGER_HIDDEN_DIM,
            num_layers=Config.TAGGER_NUM_LAYERS,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_NUM_LAYERS > 1 else 0,
            bidirectional=Config.TAGGER_BIDIRECTIONAL,
            batch_first=True,
        )

        # Output Projection Layer
        lstm_output_dim = (
            Config.TAGGER_HIDDEN_DIM * 2
            if Config.TAGGER_BIDIRECTIONAL
            else Config.TAGGER_HIDDEN_DIM
        )
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

    def forward(self, word_ids, char_ids):
        # word_ids: (Batch, Seq_Len)
        # char_ids: (Batch, Seq_Len, Char_Len)

        # Get Word Embeddings
        word_embeds = self.word_embedding(word_ids)  # (Batch, Seq, Word_Dim)

        # Get Character Features
        char_feats = self.char_encoder(char_ids)  # (Batch, Seq, Char_Dim)

        # Concatenate features
        combined = torch.cat(
            (word_embeds, char_feats), dim=2
        )  # (Batch, Seq, Total_Dim)
        combined = self.dropout(combined)

        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(combined)  # (Batch, Seq, Hidden*Dirs)
        lstm_out = self.dropout(lstm_out)

        # Project to Class Logits
        logits = self.fc(lstm_out)  # (Batch, Seq, Num_Classes)

        return logits


class EncoderRNN(nn.Module):
    """
    LSTM Encoder for Seq2Seq Model.
    """

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            input_size, Config.SEQ2SEQ_EMBED_DIM, padding_idx=0
        )
        self.lstm = nn.LSTM(Config.SEQ2SEQ_EMBED_DIM, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        # input_seq: (Batch, Seq)
        embedded = self.dropout(self.embedding(input_seq))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """
    Attention Mechanism for Seq2Seq Decoder.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (Batch, Hidden) - current decoder hidden state
        # encoder_outputs: (Batch, Seq, Hidden)

        src_len = encoder_outputs.size(1)

        # Repeat hidden state src_len times to align with encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (Batch, Seq, Hidden)

        # Calculate energy
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )  # (Batch, Seq, Hidden)

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)  # (Batch, Seq)

        return F.softmax(attention, dim=1)


class DecoderRNN(nn.Module):
    """
    LSTM Decoder with Attention for Seq2Seq Model.
    """

    def __init__(self, output_size, hidden_size, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(
            output_size, Config.SEQ2SEQ_EMBED_DIM, padding_idx=0
        )
        self.attention = Attention(hidden_size)

        self.lstm = nn.LSTM(
            Config.SEQ2SEQ_EMBED_DIM + hidden_size, hidden_size, batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        # input_step: (Batch) - indices
        # hidden, cell: (1, Batch, Hidden)

        input_step = input_step.unsqueeze(1)  # (Batch, 1)
        embedded = self.dropout(self.embedding(input_step))  # (Batch, 1, Embed)

        # Calculate attention weights using the last hidden state
        # hidden[-1] gives (Batch, Hidden)
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # (Batch, Seq)

        # Context vector (weighted sum of encoder outputs)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        )  # (Batch, 1, Hidden)

        # Concatenate embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # (Batch, 1, Embed + Hidden)

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        prediction = self.out(output.squeeze(1))  # (Batch, Output_Size)

        return prediction, hidden, cell, attn_weights


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-Sequence model with Attention for Neural Fallback.
    Used to normalize OOV tokens.
    """

    def __init__(self, num_chars):
        super(Seq2SeqModel, self).__init__()
        self.encoder = EncoderRNN(
            num_chars, Config.SEQ2SEQ_HIDDEN_DIM, Config.SEQ2SEQ_DROPOUT
        )
        self.decoder = DecoderRNN(
            num_chars, Config.SEQ2SEQ_HIDDEN_DIM, Config.SEQ2SEQ_DROPOUT
        )
        self.vocab_size = num_chars

    def forward(self, src_ids, tgt_ids=None, teacher_forcing_ratio=0.5):
        # src_ids: (Batch, Src_Len)
        # tgt_ids: (Batch, Tgt_Len) - includes SOS at start. Required for training.

        batch_size = src_ids.size(0)
        max_len = Config.SEQ2SEQ_MAX_LEN
        if tgt_ids is not None:
            max_len = tgt_ids.size(1)

        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src_ids)

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(src_ids.device)

        # Initial input is SOS
        if tgt_ids is not None:
            input_step = tgt_ids[:, 0]
        else:
            raise ValueError(
                "tgt_ids must be provided for forward pass (training). Use generate() for inference."
            )

        # Decode Loop
        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(
                input_step, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output

            # Teacher Forcing: Use actual target as next input with probability
            use_teacher_forcing = (
                True if random.random() < teacher_forcing_ratio else False
            )

            if use_teacher_forcing and tgt_ids is not None:
                input_step = tgt_ids[:, t]
            else:
                input_step = output.argmax(1)

        return outputs

    def generate(self, src_ids, sos_idx, eos_idx, max_len=None):
        """
        Inference generation method.
        Greedy decoding.
        """
        if max_len is None:
            max_len = Config.SEQ2SEQ_MAX_LEN

        batch_size = src_ids.size(0)
        encoder_outputs, hidden, cell = self.encoder(src_ids)

        # Start with SOS
        input_step = torch.tensor([sos_idx] * batch_size, device=src_ids.device)

        generated_ids = []

        for t in range(max_len):
            output, hidden, cell, _ = self.decoder(
                input_step, hidden, cell, encoder_outputs
            )
            top1 = output.argmax(1)
            generated_ids.append(top1)
            input_step = top1

        # Stack: (Seq, Batch) -> (Batch, Seq)
        generated_ids = torch.stack(generated_ids).transpose(0, 1)
        return generated_ids
