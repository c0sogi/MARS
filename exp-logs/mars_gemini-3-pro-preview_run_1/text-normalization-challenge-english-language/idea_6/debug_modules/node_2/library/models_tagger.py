import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CharCNN(nn.Module):
    """
    Character-level CNN for extracting morphological features from tokens.
    Input: (Batch, Seq_Len, Char_Len)
    Output: (Batch, Seq_Len, Filters)
    """

    def __init__(self, vocab_size, embedding_dim, filters, kernel_size):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

    def forward(self, x):
        # x: (batch, seq_len, char_len)
        batch_size, seq_len, char_len = x.size()

        # Flatten batch and seq dimensions: (batch * seq_len, char_len)
        x = x.view(-1, char_len)

        # Embed: (batch * seq_len, char_len, emb_dim)
        x = self.embedding(x)

        # Permute for Conv1d: (batch * seq_len, emb_dim, char_len)
        x = x.permute(0, 2, 1)

        # Conv: (batch * seq_len, filters, char_len)
        x = self.conv(x)
        x = torch.relu(x)

        # Global Max Pooling: (batch * seq_len, filters)
        x, _ = torch.max(x, dim=2)

        # Reshape back to sequence format: (batch, seq_len, filters)
        x = x.view(batch_size, seq_len, -1)

        return self.dropout(x)


class CRFLayer(nn.Module):
    """
    Conditional Random Field (CRF) layer for sequence tagging.
    """

    def __init__(self, num_tags):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags

        # Transition scores from j to i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        """
        Computes Negative Log Likelihood (NLL).
        Returns: (batch_size,) or scalar mean depending on implementation.
        Here returns (batch_size,) loss.
        """
        log_likelihood = self._compute_log_partition(
            emissions, mask
        ) - self._compute_score(emissions, tags, mask)
        return log_likelihood

    def _compute_score(self, emissions, tags, mask):
        """
        Computes the score of the ground truth sequence.
        """
        batch_size, seq_len, _ = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)

        # Start transition
        first_tags = tags[:, 0]
        score += self.start_transitions[first_tags]
        score += emissions[torch.arange(batch_size), 0, first_tags]

        for i in range(1, seq_len):
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]

            trans_score = self.transitions[prev_tags, curr_tags]
            emit_score = emissions[torch.arange(batch_size), i, curr_tags]

            # Add score only if mask is valid
            step_score = trans_score + emit_score
            score += step_score * mask[:, i]

        # End transition
        # Find the last valid tag index
        seq_lengths = mask.sum(dim=1).long()
        last_tags_idx = torch.clamp(seq_lengths - 1, min=0)
        last_tags = tags[torch.arange(batch_size), last_tags_idx]

        score += self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions, mask):
        """
        Computes the log partition function Z(x) using the Forward Algorithm.
        """
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize alpha with start scores + first emission
        alpha = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            # Broadcast for transition matrix operation
            # alpha: (B, T_from, 1)
            # trans: (1, T_from, T_to)
            # emit:  (B, 1, T_to)

            alpha_prev = alpha.unsqueeze(2)
            trans_score = self.transitions.unsqueeze(0)
            emit_score = emissions[:, i].unsqueeze(1)

            scores = alpha_prev + trans_score + emit_score

            # LogSumExp over 'from' dimension -> (B, T_to)
            next_alpha = torch.logsumexp(scores, dim=1)

            # Masking: if mask is 0, keep previous alpha
            valid_mask = mask[:, i].unsqueeze(1)  # (B, 1)
            alpha = valid_mask * next_alpha + (~valid_mask) * alpha

        # End transition
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        """
        Viterbi Decoding.
        """
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize
        alpha = self.start_transitions + emissions[:, 0]
        history = []

        # Identity indices for padding handling
        identity_indices = (
            torch.arange(num_tags, device=emissions.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        for i in range(1, seq_len):
            alpha_prev = alpha.unsqueeze(2)
            trans_score = self.transitions.unsqueeze(0)
            emit_score = emissions[:, i].unsqueeze(1)

            scores = alpha_prev + trans_score + emit_score

            # Max over 'from' dimension
            max_scores, best_prev_tag = torch.max(scores, dim=1)

            valid_mask = mask[:, i].unsqueeze(1)

            # Update alpha
            alpha = valid_mask * max_scores + (~valid_mask) * alpha

            # Record history
            # If mask is 0 (pad), record identity transition (k -> k)
            # so backtracking stays in the same state through padding.
            step_history = torch.where(
                valid_mask.bool(), best_prev_tag, identity_indices
            )
            history.append(step_history)

        # End transition
        alpha = alpha + self.end_transitions
        _, best_last_tag = torch.max(alpha, dim=1)

        # Backtracking
        best_tags = [best_last_tag]
        curr_tags = best_last_tag

        for i in range(len(history) - 1, -1, -1):
            hist = history[i]  # (B, T)
            prev_tags = hist.gather(1, curr_tags.unsqueeze(1)).squeeze(1)
            best_tags.append(prev_tags)
            curr_tags = prev_tags

        best_tags.reverse()
        return torch.stack(best_tags, dim=1)


class BiLSTM_CRF(nn.Module):
    """
    Bi-LSTM-CRF Tagger with optional Character CNN features.
    """

    def __init__(self, vocab_size, char_vocab_size, num_classes, class_weights=None):
        super(BiLSTM_CRF, self).__init__()

        # 1. Word Embeddings
        self.word_embedding = nn.Embedding(
            vocab_size, Config.TAGGER_EMBEDDING_DIM, padding_idx=0
        )

        # 2. Char CNN
        if Config.TAGGER_USE_CHAR_CNN:
            self.char_cnn = CharCNN(
                char_vocab_size,
                Config.TAGGER_CHAR_EMBEDDING_DIM,
                Config.TAGGER_CHAR_CNN_FILTERS,
                Config.TAGGER_CHAR_CNN_KERNEL_SIZE,
            )
            input_dim = Config.TAGGER_EMBEDDING_DIM + Config.TAGGER_CHAR_CNN_FILTERS
        else:
            self.char_cnn = None
            input_dim = Config.TAGGER_EMBEDDING_DIM

        # 3. Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_dim,
            Config.TAGGER_HIDDEN_DIM // 2,
            num_layers=Config.TAGGER_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_NUM_LAYERS > 1 else 0,
        )
        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

        # 4. Projection
        self.hidden2tag = nn.Linear(Config.TAGGER_HIDDEN_DIM, num_classes)

        # 5. CRF
        self.crf = CRFLayer(num_classes) if Config.TAGGER_USE_CRF else None

        # Class Weights Buffer
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, token_ids, char_ids, mask):
        # Word Embeddings
        embeds = self.word_embedding(token_ids)  # (B, S, E)

        # Char Features
        if self.char_cnn:
            char_feats = self.char_cnn(char_ids)  # (B, S, F)
            embeds = torch.cat([embeds, char_feats], dim=2)

        embeds = self.dropout(embeds)

        # Pack sequence
        lengths = mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)  # Safety

        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )

        lstm_out, _ = self.lstm(packed)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=token_ids.size(1)
        )

        lstm_out = self.dropout(lstm_out)

        # Emissions
        emissions = self.hidden2tag(lstm_out)  # (B, S, NumClasses)

        # Apply Class Weights (Scaling Emissions)
        if self.class_weights is not None:
            # Broadcast weights: (1, 1, C)
            emissions = emissions * self.class_weights.view(1, 1, -1)

        return emissions

    def loss(self, token_ids, char_ids, tags, mask):
        """
        Computes loss for training.
        """
        emissions = self.forward(token_ids, char_ids, mask)

        if self.crf:
            # CRF NLL Loss
            nll = self.crf(emissions, tags, mask)
            return nll.mean()
        else:
            # Cross Entropy Fallback
            active_loss = mask.view(-1) == 1
            active_logits = emissions.view(-1, self.hidden2tag.out_features)[
                active_loss
            ]
            active_labels = tags.view(-1)[active_loss]

            return F.cross_entropy(
                active_logits, active_labels, weight=self.class_weights
            )

    def decode(self, token_ids, char_ids, mask):
        """
        Predicts best tag sequence.
        """
        emissions = self.forward(token_ids, char_ids, mask)

        if self.crf:
            return self.crf.decode(emissions, mask)
        else:
            return torch.argmax(emissions, dim=2)
