import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from library.config import (
    TAGGER_EMBEDDING_DIM,
    TAGGER_CHAR_EMBEDDING_DIM,
    TAGGER_CHAR_CNN_FILTERS,
    TAGGER_CHAR_CNN_KERNEL_SIZE,
    TAGGER_HIDDEN_DIM,
    TAGGER_NUM_LAYERS,
    TAGGER_DROPOUT,
    REGEX_PATTERNS,
    SEQ2SEQ_EMBEDDING_DIM,
    SEQ2SEQ_HIDDEN_DIM,
    SEQ2SEQ_NUM_LAYERS,
    SEQ2SEQ_DROPOUT,
    SEQ2SEQ_MAX_LEN,
    SOS_TOKEN,
    EOS_TOKEN,
    DEVICE,
)


class RegexBiLSTMTagger(nn.Module):
    """
    A Bi-LSTM Tagger that fuses Word Embeddings, Character-level CNN features,
    and explicit Regex features to predict token classes.
    """

    def __init__(self, vocab_size_words, vocab_size_chars, vocab_size_classes):
        super(RegexBiLSTMTagger, self).__init__()

        # 1. Word Embeddings
        self.word_embedding = nn.Embedding(
            vocab_size_words, TAGGER_EMBEDDING_DIM, padding_idx=0
        )

        # 2. Character-level CNN
        self.char_embedding = nn.Embedding(
            vocab_size_chars, TAGGER_CHAR_EMBEDDING_DIM, padding_idx=0
        )
        self.char_conv = nn.Conv1d(
            in_channels=TAGGER_CHAR_EMBEDDING_DIM,
            out_channels=TAGGER_CHAR_CNN_FILTERS,
            kernel_size=TAGGER_CHAR_CNN_KERNEL_SIZE,
            padding=1,
        )
        self.char_dropout = nn.Dropout(TAGGER_DROPOUT)

        # 3. Regex Features
        self.num_regex_feats = len(REGEX_PATTERNS)

        # 4. Bi-LSTM Backbone
        # Input dim = Word Emb + Char CNN + Regex
        self.lstm_input_dim = (
            TAGGER_EMBEDDING_DIM + TAGGER_CHAR_CNN_FILTERS + self.num_regex_feats
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=TAGGER_HIDDEN_DIM,
            num_layers=TAGGER_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=TAGGER_DROPOUT if TAGGER_NUM_LAYERS > 1 else 0,
        )

        # 5. Classifier Head
        self.classifier_dropout = nn.Dropout(TAGGER_DROPOUT)
        self.fc = nn.Linear(
            TAGGER_HIDDEN_DIM * 2, vocab_size_classes
        )  # *2 for bidirectional

    def forward(self, word_ids, char_ids, regex_features):
        """
        Args:
            word_ids: (batch, seq_len)
            char_ids: (batch, seq_len, char_len)
            regex_features: (batch, seq_len, num_regex)
        """
        batch_size, seq_len = word_ids.size()

        # --- Word Path ---
        word_embeds = self.word_embedding(word_ids)  # (batch, seq_len, emb_dim)

        # --- Char CNN Path ---
        # Flatten batch and seq dimensions to process chars in parallel
        char_len = char_ids.size(2)
        flat_char_ids = char_ids.view(-1, char_len)  # (batch*seq, char_len)

        char_embeds = self.char_embedding(
            flat_char_ids
        )  # (batch*seq, char_len, char_emb)
        # Permute for Conv1d: (N, L, C) -> (N, C, L)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_cnn_out = self.char_conv(char_embeds)  # (batch*seq, filters, L_out)
        char_cnn_out = F.relu(char_cnn_out)
        # Max pool over the character sequence length
        # output shape: (batch*seq, filters, 1) -> squeeze -> (batch*seq, filters)
        char_pool_out = F.adaptive_max_pool1d(char_cnn_out, 1).squeeze(2)

        # Reshape back to sequence structure
        char_features = char_pool_out.view(batch_size, seq_len, TAGGER_CHAR_CNN_FILTERS)
        char_features = self.char_dropout(char_features)

        # --- Concatenation ---
        # Ensure regex_features is float
        regex_features = regex_features.float()

        combined_input = torch.cat([word_embeds, char_features, regex_features], dim=2)

        # --- LSTM ---
        lstm_out, _ = self.lstm(combined_input)  # (batch, seq_len, hidden*2)

        # --- Classifier ---
        lstm_out = self.classifier_dropout(lstm_out)
        logits = self.fc(lstm_out)  # (batch, seq_len, num_classes)

        return logits


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, dec_hid_dim) - current decoder hidden state
        # encoder_outputs: (batch, src_len, enc_hid_dim * 2)

        src_len = encoder_outputs.shape[1]

        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention scores
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class CharLSTMSeq2Seq(nn.Module):
    """
    Character-level LSTM Encoder-Decoder with Attention.
    Conditioned on the Class Embedding to guide normalization style.
    """

    def __init__(self, vocab_size_chars, vocab_size_classes, sos_idx, eos_idx):
        super(CharLSTMSeq2Seq, self).__init__()

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.vocab_size_chars = vocab_size_chars

        # --- Encoder ---
        self.enc_embedding = nn.Embedding(
            vocab_size_chars, SEQ2SEQ_EMBEDDING_DIM, padding_idx=0
        )
        self.enc_lstm = nn.LSTM(
            SEQ2SEQ_EMBEDDING_DIM,
            SEQ2SEQ_HIDDEN_DIM,
            SEQ2SEQ_NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=SEQ2SEQ_DROPOUT if SEQ2SEQ_NUM_LAYERS > 1 else 0,
        )
        self.enc_dropout = nn.Dropout(SEQ2SEQ_DROPOUT)

        # --- Class Conditioning ---
        # We embed the class and use it to initialize/condition the decoder
        self.class_embedding = nn.Embedding(vocab_size_classes, SEQ2SEQ_HIDDEN_DIM)

        # --- Decoder ---
        self.dec_embedding = nn.Embedding(
            vocab_size_chars, SEQ2SEQ_EMBEDDING_DIM, padding_idx=0
        )
        self.attention = Attention(SEQ2SEQ_HIDDEN_DIM, SEQ2SEQ_HIDDEN_DIM)

        # Decoder input: embedding + context vector
        self.dec_lstm = nn.LSTM(
            SEQ2SEQ_EMBEDDING_DIM
            + (SEQ2SEQ_HIDDEN_DIM * 2),  # Input: char_emb + weighted_enc_out
            SEQ2SEQ_HIDDEN_DIM,
            SEQ2SEQ_NUM_LAYERS,
            batch_first=True,
            dropout=SEQ2SEQ_DROPOUT if SEQ2SEQ_NUM_LAYERS > 1 else 0,
        )

        # Output projection
        self.fc_out = nn.Linear(
            SEQ2SEQ_HIDDEN_DIM + SEQ2SEQ_EMBEDDING_DIM + (SEQ2SEQ_HIDDEN_DIM * 2),
            vocab_size_chars,
        )
        self.dec_dropout = nn.Dropout(SEQ2SEQ_DROPOUT)

        # Bridge to initialize decoder hidden state from encoder final state + class embedding
        # Encoder is bidirectional (hidden*2), Decoder is unidirectional (hidden)
        self.bridge_h = nn.Linear(
            (SEQ2SEQ_HIDDEN_DIM * 2) + SEQ2SEQ_HIDDEN_DIM, SEQ2SEQ_HIDDEN_DIM
        )
        self.bridge_c = nn.Linear(
            (SEQ2SEQ_HIDDEN_DIM * 2) + SEQ2SEQ_HIDDEN_DIM, SEQ2SEQ_HIDDEN_DIM
        )

    def forward(self, src_char_ids, class_ids, trg_char_ids, teacher_forcing_ratio=0.5):
        # src_char_ids: (batch, src_len)
        # class_ids: (batch)
        # trg_char_ids: (batch, trg_len)

        batch_size = src_char_ids.shape[0]
        trg_len = trg_char_ids.shape[1]

        # --- Encoder Pass ---
        embedded = self.enc_dropout(self.enc_embedding(src_char_ids))
        encoder_outputs, (hidden, cell) = self.enc_lstm(embedded)

        # Prepare Decoder Initialization
        # hidden is (num_layers * num_directions, batch, hidden_size)
        # We take the last layer's forward and backward states
        h_fwd = hidden[-2, :, :]
        h_bwd = hidden[-1, :, :]
        c_fwd = cell[-2, :, :]
        c_bwd = cell[-1, :, :]

        # Class Embedding
        class_emb = self.class_embedding(class_ids)  # (batch, hidden)

        # Create initial decoder hidden state
        # Concat: [Enc_Fwd, Enc_Bwd, Class_Emb] -> Linear -> Dec_Init
        cat_h = torch.cat([h_fwd, h_bwd, class_emb], dim=1)
        cat_c = torch.cat([c_fwd, c_bwd, class_emb], dim=1)

        dec_hidden = torch.tanh(self.bridge_h(cat_h))
        dec_cell = torch.tanh(self.bridge_c(cat_c))

        # Replicate for num_layers
        dec_hidden = dec_hidden.unsqueeze(0).repeat(SEQ2SEQ_NUM_LAYERS, 1, 1)
        dec_cell = dec_cell.unsqueeze(0).repeat(SEQ2SEQ_NUM_LAYERS, 1, 1)

        # --- Decoder Loop ---
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size_chars).to(
            src_char_ids.device
        )

        # First input is SOS token
        input_token = trg_char_ids[:, 0]

        for t in range(1, trg_len):
            # Embed input
            dec_embedded = self.dec_dropout(self.dec_embedding(input_token)).unsqueeze(
                1
            )  # (batch, 1, emb)

            # Attention (using top layer hidden state of decoder)
            attn_weights = self.attention(
                dec_hidden[-1], encoder_outputs
            )  # (batch, src_len)

            # Weighted context
            context = torch.bmm(
                attn_weights.unsqueeze(1), encoder_outputs
            )  # (batch, 1, enc_hid*2)

            # LSTM input: concat embedding and context
            rnn_input = torch.cat((dec_embedded, context), dim=2)

            # Step
            dec_output, (dec_hidden, dec_cell) = self.dec_lstm(
                rnn_input, (dec_hidden, dec_cell)
            )

            # Prediction
            # Concatenate dec_output, dec_embedded, context for final linear layer
            prediction = self.fc_out(
                torch.cat((dec_output, dec_embedded, context), dim=2).squeeze(1)
            )

            outputs[:, t, :] = prediction

            # Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg_char_ids[:, t] if teacher_force else top1

        return outputs

    def predict(self, src_char_ids, class_ids, max_len=SEQ2SEQ_MAX_LEN):
        # Inference mode (Greedy decoding)
        self.eval()
        with torch.no_grad():
            batch_size = src_char_ids.shape[0]

            # Encoder
            embedded = self.enc_embedding(src_char_ids)
            encoder_outputs, (hidden, cell) = self.enc_lstm(embedded)

            # Init Decoder State
            h_fwd = hidden[-2, :, :]
            h_bwd = hidden[-1, :, :]
            c_fwd = cell[-2, :, :]
            c_bwd = cell[-1, :, :]
            class_emb = self.class_embedding(class_ids)

            cat_h = torch.cat([h_fwd, h_bwd, class_emb], dim=1)
            cat_c = torch.cat([c_fwd, c_bwd, class_emb], dim=1)

            dec_hidden = (
                torch.tanh(self.bridge_h(cat_h))
                .unsqueeze(0)
                .repeat(SEQ2SEQ_NUM_LAYERS, 1, 1)
            )
            dec_cell = (
                torch.tanh(self.bridge_c(cat_c))
                .unsqueeze(0)
                .repeat(SEQ2SEQ_NUM_LAYERS, 1, 1)
            )

            # Loop
            input_token = torch.tensor(
                [self.sos_idx] * batch_size, device=src_char_ids.device
            )
            generated_tokens = []

            for _ in range(max_len):
                dec_embedded = self.dec_embedding(input_token).unsqueeze(1)
                attn_weights = self.attention(dec_hidden[-1], encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                rnn_input = torch.cat((dec_embedded, context), dim=2)

                dec_output, (dec_hidden, dec_cell) = self.dec_lstm(
                    rnn_input, (dec_hidden, dec_cell)
                )
                prediction = self.fc_out(
                    torch.cat((dec_output, dec_embedded, context), dim=2).squeeze(1)
                )

                input_token = prediction.argmax(1)
                generated_tokens.append(input_token.cpu().numpy())

            # Transpose to (batch, seq_len)
            return np.array(generated_tokens).T
