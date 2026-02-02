import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from library.config import Config


class CharCNN(nn.Module):
    """
    Character-level CNN for extracting morphological features.
    """

    def __init__(self, num_chars, embed_dim, num_filters, kernel_size):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

    def forward(self, x):
        # x: (Batch * Seq, Char_Len)

        # Embed and permute for Conv1d (Batch, Channels, Length)
        x_embed = self.embedding(x)  # (B*S, C_Len, Emb)
        x_embed = x_embed.permute(0, 2, 1)  # (B*S, Emb, C_Len)

        # Convolution + ReLU
        conv_out = self.conv(x_embed)  # (B*S, Filters, C_Len)
        conv_out = F.relu(conv_out)

        # Global Max Pooling over character sequence length
        pooled, _ = torch.max(conv_out, dim=2)  # (B*S, Filters)

        return self.dropout(pooled)


class MorphEnhancedTagger(nn.Module):
    """
    Stage 1 Model: Bi-LSTM Tagger with Multi-Modal Input.
    Combines Word Embeddings, Char-CNN features, and Explicit Regex features.
    """

    def __init__(self, vocab_size, num_classes, num_chars, num_explicit_features):
        super().__init__()

        # 1. Word Embeddings
        self.word_embedding = nn.Embedding(
            vocab_size, Config.TAGGER_EMBED_DIM, padding_idx=0
        )
        self.word_dropout = nn.Dropout(Config.TAGGER_DROPOUT)

        # 2. Character CNN
        self.char_cnn = CharCNN(
            num_chars=num_chars,
            embed_dim=Config.TAGGER_CHAR_EMBED_DIM,
            num_filters=Config.TAGGER_CNN_FILTERS,
            kernel_size=Config.TAGGER_CNN_KERNEL_SIZE,
        )

        # 3. Fusion Dimension
        # Concatenating: Word Embed (256) + Char CNN (128) + Explicit Features (~25)
        input_dim = (
            Config.TAGGER_EMBED_DIM + Config.TAGGER_CNN_FILTERS + num_explicit_features
        )

        # 4. Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.TAGGER_HIDDEN_DIM,
            num_layers=Config.TAGGER_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_LAYERS > 1 else 0,
        )

        # 5. Classification Head
        self.classifier = nn.Linear(Config.TAGGER_HIDDEN_DIM * 2, num_classes)

    def forward(self, word_indices, char_indices, explicit_features):
        """
        Args:
            word_indices: (Batch, Seq)
            char_indices: (Batch, Seq, Char_Len)
            explicit_features: (Batch, Seq, Num_Features)
        """
        batch_size, seq_len = word_indices.size()

        # 1. Word Branch
        word_embeds = self.word_embedding(word_indices)  # (B, S, Emb)
        word_embeds = self.word_dropout(word_embeds)

        # 2. Character Branch
        # Reshape to (Batch * Seq, Char_Len) for CNN
        char_flat = char_indices.view(-1, char_indices.size(2))
        char_feats_flat = self.char_cnn(char_flat)  # (B*S, Filters)
        # Reshape back to (Batch, Seq, Filters)
        char_feats = char_feats_flat.view(batch_size, seq_len, -1)

        # 3. Concatenate all features
        # explicit_features is (B, S, F)
        combined_input = torch.cat([word_embeds, char_feats, explicit_features], dim=2)

        # 4. Sequence Modeling
        lstm_out, _ = self.lstm(combined_input)  # (B, S, Hidden*2)

        # 5. Prediction
        logits = self.classifier(lstm_out)  # (B, S, Num_Classes)

        return logits


# =========================================================================
# Seq2Seq Fallback Components
# =========================================================================


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (Batch, Seq)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (Batch, Hid_Dim) - Decoder hidden state
        # encoder_outputs: (Batch, Seq, Hid_Dim)

        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (B, S, H)

        # Calculate energy
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )  # (B, S, H)

        # Calculate attention scores
        attention = self.v(energy).squeeze(2)  # (B, S)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: (Batch) - single char index
        # hidden, cell: (Layers, Batch, Hid)
        # encoder_outputs: (Batch, Seq, Hid)

        input = input.unsqueeze(1)  # (Batch, 1)
        embedded = self.dropout(self.embedding(input))  # (Batch, 1, Emb)

        # Calculate attention weights using the top layer hidden state
        a = self.attention(hidden[-1], encoder_outputs)  # (Batch, Seq)
        a = a.unsqueeze(1)  # (Batch, 1, Seq)

        # Weighted sum of encoder outputs (Context Vector)
        weighted = torch.bmm(a, encoder_outputs)  # (Batch, 1, Hid)

        # RNN Input: Concat embedding and context vector
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Prediction: Concat Output, Context, Embedding
        prediction_input = torch.cat((output, weighted, embedded), dim=2).squeeze(1)
        prediction = self.fc_out(prediction_input)

        return prediction, hidden, cell


class Seq2SeqFallback(nn.Module):
    """
    Stage 2 Model: Class-Conditioned LSTM Seq2Seq with Attention.
    Used for OOV tokens where the Tagger's class prediction guides generation.
    """

    def __init__(self, char_vocab_size, num_classes):
        super().__init__()

        self.encoder = Encoder(
            input_dim=char_vocab_size,
            emb_dim=Config.SEQ2SEQ_EMBED_DIM,
            hid_dim=Config.SEQ2SEQ_HIDDEN_DIM,
            n_layers=Config.SEQ2SEQ_LAYERS,
            dropout=Config.SEQ2SEQ_DROPOUT,
        )

        self.decoder = Decoder(
            output_dim=char_vocab_size,
            emb_dim=Config.SEQ2SEQ_EMBED_DIM,
            hid_dim=Config.SEQ2SEQ_HIDDEN_DIM,
            n_layers=Config.SEQ2SEQ_LAYERS,
            dropout=Config.SEQ2SEQ_DROPOUT,
        )

        # Conditioning: Embedding for the predicted class
        self.class_embedding = nn.Embedding(num_classes, Config.SEQ2SEQ_HIDDEN_DIM)

    def forward(self, src, tgt, class_idx, teacher_forcing_ratio=0.5):
        """
        Training forward pass.
        """
        batch_size = src.shape[0]
        max_len = tgt.shape[1]
        vocab_size = self.decoder.output_dim

        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)

        # Condition: Add class embedding to Encoder's final hidden/cell states
        # This initializes the Decoder with class-aware context.
        class_embed = self.class_embedding(class_idx)  # (Batch, Hid)
        # Expand to match layers
        class_embed = class_embed.unsqueeze(0).repeat(Config.SEQ2SEQ_LAYERS, 1, 1)

        hidden = hidden + class_embed
        cell = cell + class_embed

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)

        # First input is <SOS> (assumed at tgt[:, 0])
        input = tgt[:, 0]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            # Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

    def generate(self, src, class_idx, max_len=128, sos_idx=2, eos_idx=3):
        """
        Inference forward pass (Greedy Decoding).
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]

            encoder_outputs, hidden, cell = self.encoder(src)

            # Apply Conditioning
            class_embed = self.class_embedding(class_idx)
            class_embed = class_embed.unsqueeze(0).repeat(Config.SEQ2SEQ_LAYERS, 1, 1)
            hidden = hidden + class_embed
            cell = cell + class_embed

            # Start token
            input = torch.tensor([sos_idx] * batch_size, device=src.device)

            # Store predictions
            predictions = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=src.device
            )
            predictions[:, 0] = sos_idx

            # Track finished sequences
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

            for t in range(1, max_len):
                output, hidden, cell = self.decoder(
                    input, hidden, cell, encoder_outputs
                )
                top1 = output.argmax(1)
                predictions[:, t] = top1
                input = top1

                # Check for EOS
                is_eos = top1 == eos_idx
                finished = finished | is_eos
                if finished.all():
                    break

            return predictions
