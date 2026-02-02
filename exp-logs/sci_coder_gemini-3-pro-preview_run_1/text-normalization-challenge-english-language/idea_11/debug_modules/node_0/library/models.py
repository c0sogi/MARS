import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PriorAugmentedBiLSTMTagger(nn.Module):
    """
    Penta-Hybrid Tagger combining Word, BPE, Char-CNN, Regex, and Global Prior features
    fed into a Bi-LSTM backbone.
    """

    def __init__(self, vocab_manager, regex_dim, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # 1. Word Embeddings
        word_vocab_size = len(vocab_manager.get_word_vocab())
        self.word_embed = nn.Embedding(
            word_vocab_size, Config.EMBED_DIM_WORD, padding_idx=0
        )

        # 2. BPE Embeddings
        bpe_vocab_size = len(vocab_manager.get_bpe_tokenizer())
        self.bpe_embed = nn.Embedding(
            bpe_vocab_size, Config.EMBED_DIM_BPE, padding_idx=0
        )

        # 3. Character CNN
        char_vocab_size = len(vocab_manager.get_char_vocab())
        self.char_embed = nn.Embedding(
            char_vocab_size, Config.EMBED_DIM_CHAR, padding_idx=0
        )
        self.char_cnn = nn.Conv1d(
            in_channels=Config.EMBED_DIM_CHAR,
            out_channels=Config.CHAR_CNN_FILTERS,
            kernel_size=Config.CHAR_CNN_KERNEL_SIZE,
            padding=1,
        )

        # 4. Regex Features (Direct input, no embedding needed, maybe projection?)
        # We project them to mix them before concatenation, or use raw.
        # Using raw features is fine, but a small projection can help scale.
        # Let's keep raw as per description, but ensure dtype is float.
        self.regex_dim = regex_dim

        # 5. Global Prior Features
        # Input is a probability vector of size num_classes.
        # We apply dropout directly to the input vector.
        self.prior_dropout = nn.Dropout(Config.PRIOR_DROPOUT)
        self.prior_dim = num_classes

        # Calculate Total Input Dimension
        self.input_dim = (
            Config.EMBED_DIM_WORD
            + Config.EMBED_DIM_BPE
            + Config.CHAR_CNN_FILTERS
            + self.regex_dim
            + self.prior_dim
        )

        # Backbone: Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        # Classification Head
        lstm_out_dim = (
            Config.LSTM_HIDDEN_SIZE * 2
            if Config.LSTM_BIDIRECTIONAL
            else Config.LSTM_HIDDEN_SIZE
        )
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

        self.dropout = nn.Dropout(Config.LSTM_DROPOUT)

    def forward(self, word_ids, bpe_ids, char_ids, regex_feats, prior_feats):
        """
        Args:
            word_ids: (B, Seq)
            bpe_ids: (B, Seq, Sub_Len)
            char_ids: (B, Seq, Char_Len)
            regex_feats: (B, Seq, Regex_Dim)
            prior_feats: (B, Seq, Class_Dim)
        """
        batch_size, seq_len = word_ids.size()

        # 1. Word Features
        # (B, S, Emb_Word)
        word_emb = self.word_embed(word_ids)

        # 2. BPE Features
        # (B, S, Sub, Emb_BPE)
        bpe_emb = self.bpe_embed(bpe_ids)
        # Mean pooling over subwords, ignoring padding (0)
        # Mask: (B, S, Sub, 1)
        bpe_mask = (bpe_ids != 0).float().unsqueeze(-1)
        # Sum embeddings
        bpe_sum = (bpe_emb * bpe_mask).sum(dim=2)
        # Count non-pad tokens (clamp to 1 to avoid div by zero)
        bpe_counts = bpe_mask.sum(dim=2).clamp(min=1.0)
        # (B, S, Emb_BPE)
        bpe_feat = bpe_sum / bpe_counts

        # 3. Char CNN Features
        # Flatten: (B*S, Char_Len)
        char_ids_flat = char_ids.view(-1, char_ids.size(2))
        # (B*S, Char_Len, Emb_Char)
        char_emb_flat = self.char_embed(char_ids_flat)
        # Permute for CNN: (B*S, Emb_Char, Char_Len)
        char_emb_flat = char_emb_flat.permute(0, 2, 1)
        # CNN: (B*S, Filters, Char_Len)
        cnn_out = self.char_cnn(char_emb_flat)
        # Max Pool: (B*S, Filters)
        cnn_out = F.max_pool1d(cnn_out, kernel_size=cnn_out.size(2)).squeeze(2)
        # Reshape back: (B, S, Filters)
        char_feat = cnn_out.view(batch_size, seq_len, -1)

        # 4. Regex Features
        # (B, S, Regex_Dim)
        regex_feat = regex_feats

        # 5. Prior Features
        # (B, S, Class_Dim)
        prior_feat = self.prior_dropout(prior_feats)

        # Concatenate
        # (B, S, Total_Dim)
        combined_features = torch.cat(
            [word_emb, bpe_feat, char_feat, regex_feat, prior_feat], dim=2
        )

        combined_features = self.dropout(combined_features)

        # LSTM
        lstm_out, _ = self.lstm(combined_features)

        # Classifier
        logits = self.classifier(lstm_out)

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, S, D)
        return x + self.pe[:, : x.size(1), :]


class TransformerSeq2Seq(nn.Module):
    """
    Transformer Encoder-Decoder for OOV normalization.
    Conditioned on the predicted class ID.
    """

    def __init__(self, vocab_manager, num_classes):
        super().__init__()

        self.char_vocab = vocab_manager.get_char_vocab()
        vocab_size = len(self.char_vocab)
        self.pad_idx = self.char_vocab["<pad>"]
        self.sos_idx = self.char_vocab["<sos>"]
        self.eos_idx = self.char_vocab["<eos>"]

        self.d_model = Config.SEQ2SEQ_EMBED_DIM

        # Embeddings
        self.char_embed = nn.Embedding(
            vocab_size, self.d_model, padding_idx=self.pad_idx
        )
        self.class_embed = nn.Embedding(num_classes, self.d_model)
        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=Config.MAX_SEQ2SEQ_LEN + 2
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=Config.SEQ2SEQ_HEADS,
            num_encoder_layers=Config.SEQ2SEQ_LAYERS,
            num_decoder_layers=Config.SEQ2SEQ_LAYERS,
            dim_feedforward=Config.SEQ2SEQ_HIDDEN_DIM,
            dropout=Config.SEQ2SEQ_DROPOUT,
            batch_first=True,
        )

        # Output Head
        self.fc_out = nn.Linear(self.d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, class_ids):
        """
        Args:
            src_ids: (B, Src_Len)
            tgt_ids: (B, Tgt_Len) - Includes <sos> and <eos>
            class_ids: (B)
        """
        # 1. Prepare Source Embedding with Class Conditioning
        # (B, Src_Len, D)
        src_emb = self.char_embed(src_ids) * math.sqrt(self.d_model)

        # Prepend Class Embedding as a token at the start of source
        # (B, 1, D)
        cls_emb = self.class_embed(class_ids).unsqueeze(1) * math.sqrt(self.d_model)

        # (B, Src_Len + 1, D)
        src_emb = torch.cat([cls_emb, src_emb], dim=1)
        src_emb = self.pos_encoder(src_emb)

        # 2. Prepare Target Embedding
        # (B, Tgt_Len, D)
        tgt_emb = self.char_embed(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 3. Masks
        # Padding Mask for Source (True where padded)
        # We need to account for the prepended class token (never padded)
        # src_ids: (B, S), mask: (B, S)
        src_key_padding_mask = src_ids == self.pad_idx
        # Add False for the class token column
        # (B, 1)
        cls_mask = torch.zeros(
            (src_ids.size(0), 1), dtype=torch.bool, device=src_ids.device
        )
        # (B, S+1)
        src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # Padding Mask for Target
        tgt_key_padding_mask = tgt_ids == self.pad_idx

        # Causal Mask for Target (prevent looking ahead)
        tgt_seq_len = tgt_ids.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            src_ids.device
        )

        # 4. Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        # 5. Output Logits
        logits = self.fc_out(output)
        return logits

    def generate(self, src_ids, class_ids, max_len=Config.MAX_SEQ2SEQ_LEN):
        """
        Greedy decoding for inference.
        Args:
            src_ids: (B, Src_Len)
            class_ids: (B)
        """
        batch_size = src_ids.size(0)
        device = src_ids.device

        # 1. Encode Source
        src_emb = self.char_embed(src_ids) * math.sqrt(self.d_model)
        cls_emb = self.class_embed(class_ids).unsqueeze(1) * math.sqrt(self.d_model)
        src_emb = torch.cat([cls_emb, src_emb], dim=1)
        src_emb = self.pos_encoder(src_emb)

        src_key_padding_mask = src_ids == self.pad_idx
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # 2. Decode Loop
        # Start with <sos>
        ys = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_len):
            tgt_emb = self.char_embed(ys) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)

            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )

            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Get logits for the last token
            prob = self.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Append
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Check for EOS
            is_eos = next_word == self.eos_idx
            finished = finished | is_eos

            if finished.all():
                break

        # Remove <sos>
        return ys[:, 1:]
