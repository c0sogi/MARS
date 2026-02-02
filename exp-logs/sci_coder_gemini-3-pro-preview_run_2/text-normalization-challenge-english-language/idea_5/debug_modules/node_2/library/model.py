import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config
from library.utils import get_logger

logger = get_logger()


class CRFLayer(nn.Module):
    """
    Conditional Random Field (CRF) layer for sequence labeling.
    Implements the Forward Algorithm for loss calculation and Viterbi Algorithm for decoding.
    """

    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # Transitions[i, j] is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None):
        """
        Computes the Negative Log Likelihood (NLL) of the given sequence of tags.

        Args:
            emissions: (Batch, Seq, NumTags)
            tags: (Batch, Seq)
            mask: (Batch, Seq) - 1 for valid tokens, 0 for padding

        Returns:
            loss: Scalar (mean NLL over batch)
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)  # (Seq, Batch, K)
            tags = tags.transpose(0, 1)  # (Seq, Batch)
            mask = mask.transpose(0, 1)  # (Seq, Batch)

        seq_length, batch_size = tags.shape

        # ====================================================
        # 1. Compute Score of the Ground Truth Sequence (Numerator)
        # ====================================================
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score from prev tag to current tag
            trans_score = self.transitions[tags[i], tags[i - 1]]
            # Emission score for current tag
            emit_score = emissions[i, torch.arange(batch_size), tags[i]]

            # Only add scores if the current position is valid (mask == 1)
            score += (trans_score + emit_score) * mask[i]

        # Add end transition score
        # We take the tag at the last valid position for each batch element
        seq_lengths = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_lengths, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        # ====================================================
        # 2. Compute Partition Function Z (Denominator) via Forward Algorithm
        # ====================================================
        # alpha[b, k] = log-prob of reaching state k at current step for batch b
        alpha = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast for efficient computation:
            # alpha_t: (Batch, From, 1) -> unsqueeze dim 2
            # trans:   (1, From, To)    -> unsqueeze dim 0 (Wait, self.transitions is To, From)
            # Let's align dimensions carefully.

            # We want: next_alpha[b, to] = logsumexp(alpha[b, from] + trans[to, from]) + emit[b, to]

            # alpha: (Batch, From) -> (Batch, 1, From)
            alpha_t = alpha.unsqueeze(1)

            # transitions: (To, From) -> (1, To, From)
            trans_t = self.transitions.unsqueeze(0)

            # scores: (Batch, To, From) represents all possible transitions to 'To'
            scores = alpha_t + trans_t

            # LogSumExp over 'From' dimension -> (Batch, To)
            next_alpha = torch.logsumexp(scores, dim=2)

            # Add emissions for the current step: (Batch, To)
            next_alpha += emissions[i]

            # Masking: If mask[i] is 0, we must carry over the previous alpha values
            # because the sequence hasn't advanced in reality (padding).
            mask_t = mask[i].unsqueeze(1)
            alpha = mask_t * next_alpha + (1 - mask_t) * alpha

        # Add end transitions to the final alpha values
        alpha += self.end_transitions

        # Final Z is logsumexp over all possible last tags
        Z = torch.logsumexp(alpha, dim=1)

        # NLL = Z - Score
        return torch.mean(Z - score)

    def decode(self, emissions, mask=None):
        """
        Finds the most likely sequence of tags using the Viterbi Algorithm.

        Returns:
            List[List[int]]: Best tag sequence for each batch element.
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.uint8, device=emissions.device
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size, num_tags = emissions.shape

        # Initialize path scores
        path_score = self.start_transitions + emissions[0]
        best_paths = []  # To store backpointers

        for i in range(1, seq_length):
            # path_score: (Batch, From) -> (Batch, 1, From)
            score_t = path_score.unsqueeze(1)
            # transitions: (To, From) -> (1, To, From)
            trans_t = self.transitions.unsqueeze(0)

            # scores: (Batch, To, From)
            scores = score_t + trans_t

            # Max over 'From' -> (Batch, To)
            best_score, best_from = torch.max(scores, dim=2)

            # Add emissions
            best_score += emissions[i]

            # Handle masking
            mask_t = mask[i].unsqueeze(1)
            path_score = mask_t * best_score + (1 - mask_t) * path_score

            best_paths.append(best_from)

        # Add end transitions
        path_score += self.end_transitions

        # Get the best tag at the end of the sequence
        _, best_last_tag = torch.max(path_score, dim=1)

        # Backtracking
        batch_paths = []
        seq_lengths = mask.long().sum(dim=0)

        # Convert list of tensors to a single tensor for easier indexing
        # Shape: (Seq-1, Batch, NumTags)
        best_paths_tensor = torch.stack(best_paths)

        for b in range(batch_size):
            length = seq_lengths[b].item()
            if length == 0:
                batch_paths.append([])
                continue

            # The tag at 'length-1' is best_last_tag[b]
            path = [best_last_tag[b].item()]
            curr_tag = best_last_tag[b]

            # We backtrack from step (length-1) down to 1
            # The backpointer for step i is stored at best_paths[i-1]
            for i in range(length - 1, 0, -1):
                prev_tag = best_paths_tensor[i - 1, b, curr_tag]
                path.append(prev_tag.item())
                curr_tag = prev_tag

            # Reverse to get correct order
            batch_paths.append(path[::-1])

        return batch_paths


class TransformerCRF(nn.Module):
    """
    Main model class combining a Transformer backbone with a CRF layer.
    """

    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(config.hidden_size, Config.NUM_LABELS)
        self.crf = CRFLayer(Config.NUM_LABELS, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: (Batch, Seq)
            attention_mask: (Batch, Seq)
            labels: (Batch, Seq) - Optional, for training.

        Returns:
            If labels is provided: Returns scalar loss.
            If labels is None: Returns list of predicted tag IDs.
        """
        # 1. Transformer Encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # 2. Emission Scores
        emissions = self.hidden2tag(sequence_output)

        if labels is not None:
            # 3. Training: Compute CRF Loss

            # Preprocessing: The dataset uses -100 for subwords to ignore them.
            # For CRF, we need a dense sequence of labels.
            # We forward-fill the -100 values with the previous valid label.
            # This teaches the model to predict the word's label for all its subwords.

            filled_labels = labels.clone()

            # Iterate over batch to fill gaps
            # Note: This loop is fast enough for typical batch sizes (32-64) and seq len (256)
            for b in range(filled_labels.size(0)):
                last_valid = 0  # Default to PLAIN (0) if start is invalid
                for t in range(filled_labels.size(1)):
                    if filled_labels[b, t] != -100:
                        last_valid = filled_labels[b, t]
                    else:
                        filled_labels[b, t] = last_valid

            # Compute NLL loss
            loss = self.crf(emissions, filled_labels, mask=attention_mask)
            return loss

        else:
            # 4. Inference: Decode Best Sequence
            tags = self.crf.decode(emissions, mask=attention_mask)
            return tags
