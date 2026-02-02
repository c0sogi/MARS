import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FeatureDropout(nn.Module):
    """
    Applies dropout to entire feature vectors (channels) rather than individual elements.
    Used to randomly mask out Priors and Regex features during training to force
    contextual learning.
    """

    def __init__(self, p=0.5):
        super(FeatureDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        # x shape: (batch, seq_len, feature_dim)
        # We want to create a mask of shape (batch, seq_len, 1) to drop the whole feature vector
        # for a specific token, or (batch, 1, 1) to drop for the whole sequence?
        # Usually, we drop per token to simulate missing info.
        mask = torch.empty(x.shape[0], x.shape[1], 1, device=x.device).bernoulli_(
            1 - self.p
        )

        # Scale the output to maintain expected value
        return x * mask / (1 - self.p)


class PriorInformedTagger(nn.Module):
    """
    Stage 1: Classification Model.
    Combines Word, BPE, Char-CNN, Regex, and Prior features into a Bi-LSTM.
    """

    def __init__(self, vocab_words, vocab_classes, bpe_vocab_size, vocab_chars):
        super(PriorInformedTagger, self).__init__()

        self.num_classes = len(vocab_classes)

        # 1. Word Embeddings
        self.word_embedding = nn.Embedding(
            num_embeddings=len(vocab_words),
            embedding_dim=Config.TAGGER_EMBED_DIM,
            padding_idx=0,
        )

        # 2. BPE Embeddings
        self.bpe_embedding = nn.Embedding(
            num_embeddings=bpe_vocab_size,
            embedding_dim=Config.TAGGER_BPE_EMBED_DIM,
            padding_idx=0,
        )

        # 3. Character CNN
        self.char_embedding = nn.Embedding(
            num_embeddings=len(vocab_chars),
            embedding_dim=Config.TAGGER_CHAR_EMBED_DIM,
            padding_idx=0,
        )
        self.char_cnn = nn.Conv1d(
            in_channels=Config.TAGGER_CHAR_EMBED_DIM,
            out_channels=Config.TAGGER_CHAR_CNN_FILTERS,
            kernel_size=Config.TAGGER_CHAR_CNN_KERNEL_SIZE,
            padding=1,
        )

        # 4. Feature Dropout for Priors and Regex
        self.feature_dropout = FeatureDropout(p=Config.TAGGER_FEATURE_DROPOUT)

        # Calculate total input dimension for LSTM
        # Word + BPE + CharCNN + Regex + Priors
        self.input_dim = (
            Config.TAGGER_EMBED_DIM
            + Config.TAGGER_BPE_EMBED_DIM
            + Config.TAGGER_CHAR_CNN_FILTERS
            + Config.NUM_REGEX_FEATURES
            + self.num_classes  # Prior vector size
        )

        self.projection = nn.Linear(self.input_dim, Config.TAGGER_HIDDEN_DIM)
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

        # 5. Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=Config.TAGGER_HIDDEN_DIM,
            hidden_size=Config.TAGGER_HIDDEN_DIM,
            num_layers=Config.TAGGER_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_NUM_LAYERS > 1 else 0,
        )

        # 6. Classification Head
        self.classifier = nn.Linear(Config.TAGGER_HIDDEN_DIM * 2, self.num_classes)

    def forward(self, word_ids, bpe_ids, char_ids, regex_features, prior_features):
        """
        Args:
            word_ids: (batch, seq)
            bpe_ids: (batch, seq, bpe_len)
            char_ids: (batch, seq, char_len)
            regex_features: (batch, seq, num_regex)
            prior_features: (batch, seq, num_classes)
        """
        batch_size, seq_len = word_ids.size()

        # 1. Word Features
        word_emb = self.word_embedding(word_ids)  # (batch, seq, word_dim)

        # 2. BPE Features (Mean Pooling)
        # bpe_ids: (batch, seq, bpe_len) -> flatten to embed -> reshape -> mean
        bpe_flat = bpe_ids.view(-1, bpe_ids.size(-1))  # (batch*seq, bpe_len)
        bpe_emb_flat = self.bpe_embedding(bpe_flat)  # (batch*seq, bpe_len, bpe_dim)
        # Mask padding (0) for correct mean
        bpe_mask = (bpe_flat != 0).unsqueeze(-1).float()  # (batch*seq, bpe_len, 1)
        bpe_sum = (bpe_emb_flat * bpe_mask).sum(dim=1)
        bpe_count = bpe_mask.sum(dim=1).clamp(min=1.0)
        bpe_pooled = bpe_sum / bpe_count  # (batch*seq, bpe_dim)
        bpe_features = bpe_pooled.view(batch_size, seq_len, -1)

        # 3. Char CNN Features
        # char_ids: (batch, seq, char_len)
        char_flat = char_ids.view(-1, char_ids.size(-1))  # (batch*seq, char_len)
        char_emb_flat = self.char_embedding(
            char_flat
        )  # (batch*seq, char_len, char_dim)
        # Permute for Conv1d: (N, C, L)
        char_emb_perm = char_emb_flat.permute(0, 2, 1)
        cnn_out = self.char_cnn(char_emb_perm)  # (batch*seq, filters, L_out)
        # Max Pooling over time
        cnn_pooled, _ = torch.max(cnn_out, dim=2)  # (batch*seq, filters)
        char_features = cnn_pooled.view(batch_size, seq_len, -1)

        # 4. Explicit Features (Regex + Priors) with Dropout
        regex_features = self.feature_dropout(regex_features)
        prior_features = self.feature_dropout(prior_features)

        # 5. Concatenation
        combined = torch.cat(
            [word_emb, bpe_features, char_features, regex_features, prior_features],
            dim=-1,
        )

        # Projection
        projected = self.dropout(F.relu(self.projection(combined)))

        # LSTM
        lstm_out, _ = self.lstm(projected)  # (batch, seq, hidden*2)

        # Classifier
        logits = self.classifier(lstm_out)  # (batch, seq, num_classes)

        return logits


class Attention(nn.Module):
    """
    Luong-style General Attention.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(
            hidden_dim, 1, bias=False
        )  # Not used in 'General', used in 'Concat'
        # For 'General' attention score = decoder_hidden * W * encoder_output
        # Here we implement a simpler Dot-Product attention with a linear transform on query:
        # score = (W_a * decoder_hidden)^T * encoder_output
        self.W_a = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim) - Decoder state
        # encoder_outputs: (batch, src_len, hidden_dim)

        # Calculate energies
        # query = self.W_a(hidden).unsqueeze(2) # (batch, hidden, 1)
        # scores = torch.bmm(encoder_outputs, query).squeeze(2) # (batch, src_len)

        # Let's use standard dot product for simplicity and stability if dims match
        # Or concat attention. Let's use Concat as it's robust.
        # Score = v^T * tanh(W [hidden; encoder_output])

        src_len = encoder_outputs.shape[1]

        # Repeat hidden state src_len times
        hidden_expanded = hidden.unsqueeze(1).repeat(
            1, src_len, 1
        )  # (batch, src_len, hidden)

        # Concat
        combined = torch.cat(
            (hidden_expanded, encoder_outputs), dim=2
        )  # (batch, src_len, 2*hidden)

        # Energy
        # We need a layer for 2*hidden -> hidden
        # We'll define it dynamically or assume dimensions.
        # Let's stick to the simpler "General" attention: score(h_t, h_s) = h_t^T W h_s

        query = self.W_a(hidden).unsqueeze(2)  # (batch, hidden, 1)
        scores = torch.bmm(encoder_outputs, query).squeeze(2)  # (batch, src_len)

        attn_weights = F.softmax(scores, dim=1)  # (batch, src_len)

        # Context vector
        # (batch, 1, src_len) bmm (batch, src_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights


class CharLSTMSeq2Seq(nn.Module):
    """
    Stage 2: Fallback Generation Model.
    Encoder-Decoder LSTM with Attention, conditioned on Class Embedding.
    """

    def __init__(self, vocab_chars, vocab_classes):
        super(CharLSTMSeq2Seq, self).__init__()

        self.vocab_size = len(vocab_chars)
        self.class_vocab_size = len(vocab_classes)

        # Embeddings
        self.char_embedding = nn.Embedding(
            self.vocab_size, Config.SEQ2SEQ_EMBED_DIM, padding_idx=0
        )
        self.class_embedding = nn.Embedding(
            self.class_vocab_size, Config.SEQ2SEQ_EMBED_DIM
        )

        # Encoder (Bidirectional)
        self.encoder = nn.LSTM(
            input_size=Config.SEQ2SEQ_EMBED_DIM,
            hidden_size=Config.SEQ2SEQ_HIDDEN_DIM,
            num_layers=Config.SEQ2SEQ_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.SEQ2SEQ_DROPOUT if Config.SEQ2SEQ_NUM_LAYERS > 1 else 0,
        )

        # Decoder (Unidirectional)
        # Input to decoder is Char Emb + Class Emb
        self.decoder_input_dim = Config.SEQ2SEQ_EMBED_DIM + Config.SEQ2SEQ_EMBED_DIM

        self.decoder = nn.LSTM(
            input_size=self.decoder_input_dim,
            hidden_size=Config.SEQ2SEQ_HIDDEN_DIM,
            num_layers=Config.SEQ2SEQ_NUM_LAYERS,
            batch_first=True,
            dropout=Config.SEQ2SEQ_DROPOUT if Config.SEQ2SEQ_NUM_LAYERS > 1 else 0,
        )

        # Attention
        self.attention = Attention(Config.SEQ2SEQ_HIDDEN_DIM)

        # Output Projection
        # Input is [DecoderHidden; Context]
        self.out = nn.Linear(Config.SEQ2SEQ_HIDDEN_DIM * 2, self.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(Config.SEQ2SEQ_DROPOUT)

    def forward(self, src_char_ids, tgt_char_ids, class_id, teacher_forcing_ratio=0.5):
        """
        Forward pass for training.
        """
        batch_size = src_char_ids.size(0)
        tgt_len = tgt_char_ids.size(1)

        # --- Encoder ---
        src_emb = self.dropout(self.char_embedding(src_char_ids))
        encoder_outputs, (hidden, cell) = self.encoder(src_emb)

        # Handle Bidirectional Encoder -> Unidirectional Decoder
        # Sum forward and backward hidden states
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        hidden = hidden.view(
            Config.SEQ2SEQ_NUM_LAYERS, 2, batch_size, Config.SEQ2SEQ_HIDDEN_DIM
        )
        hidden = torch.sum(hidden, dim=1)  # (num_layers, batch, hidden_dim)

        cell = cell.view(
            Config.SEQ2SEQ_NUM_LAYERS, 2, batch_size, Config.SEQ2SEQ_HIDDEN_DIM
        )
        cell = torch.sum(cell, dim=1)

        # Encoder outputs: (batch, seq, hidden*2) -> Sum to match decoder hidden dim for attention
        encoder_outputs = encoder_outputs.view(
            batch_size, -1, 2, Config.SEQ2SEQ_HIDDEN_DIM
        )
        encoder_outputs = torch.sum(encoder_outputs, dim=2)  # (batch, seq, hidden)

        # --- Decoder ---
        # Get Class Embedding to condition generation
        class_emb = self.class_embedding(class_id)  # (batch, emb_dim)

        # Initial input is SOS token (assumed index 2 based on DataProcessing)
        decoder_input = torch.tensor([2] * batch_size, device=src_char_ids.device)

        outputs = []

        for t in range(tgt_len - 1):  # -1 because we don't predict after the last token
            # Embed current char
            char_emb = self.char_embedding(decoder_input)  # (batch, emb_dim)

            # Concatenate with Class Embedding (Conditioning)
            # (batch, emb_dim + emb_dim)
            rnn_input = torch.cat([char_emb, class_emb], dim=1).unsqueeze(1)

            # LSTM Step
            decoder_output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))

            # Attention
            # Use the top layer hidden state for attention
            last_layer_hidden = hidden[-1]
            context, _ = self.attention(last_layer_hidden, encoder_outputs)

            # Combine Decoder Output and Context
            # decoder_output: (batch, 1, hidden)
            output_concat = torch.cat([decoder_output.squeeze(1), context], dim=1)

            # Prediction
            prediction = self.out(output_concat)  # (batch, vocab_size)
            outputs.append(prediction.unsqueeze(1))

            # Teacher Forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt_char_ids[:, t + 1]  # Next target char
            else:
                decoder_input = prediction.argmax(dim=1)

        return torch.cat(outputs, dim=1)

    def predict(self, src_char_ids, class_id, max_len=Config.SEQ2SEQ_MAX_OUTPUT_LEN):
        """
        Greedy decoding for inference.
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_char_ids.size(0)

            # Encoder
            src_emb = self.char_embedding(src_char_ids)
            encoder_outputs, (hidden, cell) = self.encoder(src_emb)

            # Reduce Bidirectional
            hidden = hidden.view(
                Config.SEQ2SEQ_NUM_LAYERS, 2, batch_size, Config.SEQ2SEQ_HIDDEN_DIM
            )
            hidden = torch.sum(hidden, dim=1)
            cell = cell.view(
                Config.SEQ2SEQ_NUM_LAYERS, 2, batch_size, Config.SEQ2SEQ_HIDDEN_DIM
            )
            cell = torch.sum(cell, dim=1)

            encoder_outputs = encoder_outputs.view(
                batch_size, -1, 2, Config.SEQ2SEQ_HIDDEN_DIM
            )
            encoder_outputs = torch.sum(encoder_outputs, dim=2)

            # Class Conditioning
            class_emb = self.class_embedding(class_id)

            # Start Token
            decoder_input = torch.tensor([2] * batch_size, device=src_char_ids.device)

            generated_ids = []

            for _ in range(max_len):
                char_emb = self.char_embedding(decoder_input)
                rnn_input = torch.cat([char_emb, class_emb], dim=1).unsqueeze(1)

                decoder_output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))

                last_layer_hidden = hidden[-1]
                context, _ = self.attention(last_layer_hidden, encoder_outputs)

                output_concat = torch.cat([decoder_output.squeeze(1), context], dim=1)
                prediction = self.out(output_concat)

                predicted_id = prediction.argmax(dim=1)
                generated_ids.append(predicted_id.unsqueeze(1))

                # Update input
                decoder_input = predicted_id

                # Stop if all batches predicted EOS (3) - Simplifying assumption: just run to max_len or handle externally
                # Handling variable length stop in batch is complex without masking,
                # we'll just generate fixed length and truncate at EOS in post-processing.

            return torch.cat(generated_ids, dim=1)
