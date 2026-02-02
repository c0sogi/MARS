import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CharCNN(nn.Module):
    """
    Character-level CNN for extracting morphological features from tokens.
    """

    def __init__(self, num_chars, embed_dim, filters, kernel_size):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'Same' padding
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Word_Len)
        b, s, w = x.shape
        x = x.view(b * s, w)  # Flatten batch and seq dimensions

        # Embed: (B*S, W, Dim)
        emb = self.embedding(x)

        # Permute for Conv1d: (B*S, Dim, W)
        emb = emb.permute(0, 2, 1)

        # Conv: (B*S, Filters, W)
        conv_out = self.conv(emb)
        conv_out = F.relu(conv_out)

        # Global Max Pool over word length: (B*S, Filters)
        pooled, _ = torch.max(conv_out, dim=2)

        pooled = self.dropout(pooled)

        # Reshape back to (Batch, Seq_Len, Filters)
        return pooled.view(b, s, -1)


class PentaHybridTagger(nn.Module):
    """
    Bi-LSTM Tagger utilizing 5 input feature types:
    1. Word Embeddings
    2. Character CNN Features
    3. BPE Subword Embeddings
    4. Explicit Regex Features
    5. Global Prior Probabilities
    """

    def __init__(
        self, num_words, num_chars, num_bpe, num_classes, regex_dim=None, prior_dim=None
    ):
        super(PentaHybridTagger, self).__init__()

        # Hyperparameters from Config
        self.word_dim = Config.WORD_EMBED_DIM
        self.char_dim = Config.CHAR_EMBED_DIM
        self.bpe_dim = Config.BPE_EMBED_DIM
        self.cnn_filters = Config.CHAR_CNN_FILTERS
        self.cnn_kernel = Config.CHAR_CNN_KERNEL_SIZE
        self.hidden_dim = Config.LSTM_HIDDEN_DIM
        self.num_layers = Config.LSTM_LAYERS
        self.dropout_val = Config.LSTM_DROPOUT
        self.feature_dropout_val = Config.FEATURE_DROPOUT

        self.regex_dim = regex_dim if regex_dim is not None else Config.REGEX_DIM
        self.prior_dim = prior_dim if prior_dim is not None else Config.PRIOR_DIM

        # 1. Word Embeddings
        self.word_embedding = nn.Embedding(num_words, self.word_dim, padding_idx=0)

        # 2. Char CNN
        self.char_cnn = CharCNN(
            num_chars, self.char_dim, self.cnn_filters, self.cnn_kernel
        )

        # 3. BPE Embeddings
        self.bpe_embedding = nn.Embedding(num_bpe, self.bpe_dim, padding_idx=0)

        # 4. Feature Dropout (Regularization for Regex/Priors)
        self.feature_dropout = nn.Dropout(self.feature_dropout_val)

        # Calculate Total Input Dimension
        # Word + CharCNN + BPE + Regex + Prior
        self.input_dim = (
            self.word_dim
            + self.cnn_filters
            + self.bpe_dim
            + self.regex_dim
            + self.prior_dim
        )

        # 5. Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_val if self.num_layers > 1 else 0,
        )

        # 6. Classification Head
        self.classifier = nn.Linear(self.hidden_dim * 2, num_classes)

    def forward(self, word_ids, char_ids, bpe_ids, regex_feats, prior_feats):
        """
        Args:
            word_ids: (Batch, Seq)
            char_ids: (Batch, Seq, Word_Len)
            bpe_ids: (Batch, Seq, BPE_Len)
            regex_feats: (Batch, Seq, Regex_Dim)
            prior_feats: (Batch, Seq, Prior_Dim)
        Returns:
            logits: (Batch, Seq, Num_Classes)
        """
        # 1. Word Features
        word_emb = self.word_embedding(word_ids)  # (B, S, Word_Dim)

        # 2. Char Features
        char_feat = self.char_cnn(char_ids)  # (B, S, Filters)

        # 3. BPE Features (Mean Pooling with Masking)
        bpe_emb = self.bpe_embedding(bpe_ids)  # (B, S, Sub, BPE_Dim)
        mask = (bpe_ids != 0).unsqueeze(-1).float()  # (B, S, Sub, 1)
        sum_emb = torch.sum(bpe_emb * mask, dim=2)  # (B, S, BPE_Dim)
        count = torch.sum(mask, dim=2).clamp(min=1)  # (B, S, 1)
        bpe_feat = sum_emb / count

        # 4. Explicit Features (Regex + Prior)
        # Apply dropout to force model to learn from context/morphology too
        regex_feat = self.feature_dropout(regex_feats)
        prior_feat = self.feature_dropout(prior_feats)

        # Concatenate all features
        combined = torch.cat(
            [word_emb, char_feat, bpe_feat, regex_feat, prior_feat], dim=2
        )

        # LSTM
        lstm_out, _ = self.lstm(combined)  # (B, S, 2*Hidden)

        # Classify
        logits = self.classifier(lstm_out)  # (B, S, Num_Classes)

        return logits


class Attention(nn.Module):
    """
    Bahdanau-style Attention mechanism.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (Batch, Hidden) -> Decoder hidden state at time t
        # encoder_outputs: (Batch, Seq, Hidden)

        seq_len = encoder_outputs.size(1)

        # Repeat hidden state seq_len times
        h_reshaped = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (B, Seq, Hidden)

        # Energy calculation
        energy = torch.tanh(
            self.attn(torch.cat((h_reshaped, encoder_outputs), dim=2))
        )  # (B, Seq, Hidden)
        attention = self.v(energy).squeeze(2)  # (B, Seq)

        return F.softmax(attention, dim=1)


class CharLSTMSeq2Seq(nn.Module):
    """
    Character-level LSTM Seq2Seq model with Attention.
    Conditioned on the Class Embedding of the token.
    """

    def __init__(self, num_chars, num_classes):
        super(CharLSTMSeq2Seq, self).__init__()

        self.embed_dim = Config.SEQ2SEQ_EMBED_DIM
        self.hidden_dim = Config.SEQ2SEQ_HIDDEN_DIM
        self.dropout_val = Config.SEQ2SEQ_DROPOUT

        # Encoder
        self.encoder_embedding = nn.Embedding(num_chars, self.embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True)
        self.encoder_dropout = nn.Dropout(self.dropout_val)

        # Decoder
        # Condition on Class: We append class embedding to char embedding
        self.class_embedding = nn.Embedding(num_classes, self.embed_dim)
        self.decoder_embedding = nn.Embedding(num_chars, self.embed_dim, padding_idx=0)

        # Input to Decoder LSTM: Char_Emb + Class_Emb
        self.decoder_lstm = nn.LSTM(
            self.embed_dim * 2, self.hidden_dim, batch_first=True
        )

        # Attention
        self.attention = Attention(self.hidden_dim)

        # Output Projection
        # Concatenate [Hidden_t; Context_t] -> Linear -> Vocab
        self.out = nn.Linear(self.hidden_dim * 2, num_chars)

    def forward(self, src_ids, tgt_ids, class_id, teacher_forcing_ratio=0.5):
        """
        Training forward pass.
        Args:
            src_ids: (Batch, Src_Len)
            tgt_ids: (Batch, Tgt_Len) - includes SOS and EOS
            class_id: (Batch,)
        """
        batch_size = src_ids.size(0)
        tgt_len = tgt_ids.size(1)
        vocab_size = self.decoder_embedding.num_embeddings

        # Encoder
        enc_emb = self.encoder_dropout(self.encoder_embedding(src_ids))
        enc_outputs, (hidden, cell) = self.encoder_lstm(enc_emb)
        # enc_outputs: (B, Src, Hidden)
        # hidden, cell: (1, B, Hidden)

        # Prepare Class Embedding (Static context)
        class_emb = self.class_embedding(class_id)  # (B, Emb_Dim)

        # Decoder Init
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src_ids.device)

        # First input is SOS (index 0 of tgt_ids)
        input_token = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            # Embed input char
            char_emb = self.decoder_embedding(input_token)  # (B, Emb)

            # Combine with Class Condition
            dec_input = torch.cat([char_emb, class_emb], dim=1).unsqueeze(
                1
            )  # (B, 1, Emb*2)

            # LSTM Step
            dec_output, (hidden, cell) = self.decoder_lstm(dec_input, (hidden, cell))
            # dec_output: (B, 1, Hidden)

            # Attention
            # Use last hidden state (squeeze layer dim)
            attn_weights = self.attention(hidden[-1], enc_outputs)  # (B, Src)
            context = attn_weights.unsqueeze(1).bmm(enc_outputs)  # (B, 1, Hidden)

            # Output Projection
            concat_out = torch.cat((dec_output, context), dim=2)  # (B, 1, Hidden*2)
            prediction = self.out(concat_out).squeeze(1)  # (B, Vocab)

            outputs[:, t, :] = prediction

            # Teacher Forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt_ids[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs

    def generate(self, src_ids, class_id, max_len=128, sos_token_id=2, eos_token_id=3):
        """
        Inference generation.
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_ids.size(0)

            enc_emb = self.encoder_embedding(src_ids)
            enc_outputs, (hidden, cell) = self.encoder_lstm(enc_emb)

            class_emb = self.class_embedding(class_id)

            input_token = torch.tensor(
                [sos_token_id] * batch_size, device=src_ids.device
            )

            generated_ids = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)

            for _ in range(max_len):
                char_emb = self.decoder_embedding(input_token)
                dec_input = torch.cat([char_emb, class_emb], dim=1).unsqueeze(1)

                dec_output, (hidden, cell) = self.decoder_lstm(
                    dec_input, (hidden, cell)
                )

                attn_weights = self.attention(hidden[-1], enc_outputs)
                context = attn_weights.unsqueeze(1).bmm(enc_outputs)

                concat_out = torch.cat((dec_output, context), dim=2)
                prediction = self.out(concat_out).squeeze(1)

                predicted_id = prediction.argmax(1)
                generated_ids.append(predicted_id)

                input_token = predicted_id

                # Check EOS
                finished |= predicted_id == eos_token_id
                if finished.all():
                    break

            return torch.stack(generated_ids, dim=1)
