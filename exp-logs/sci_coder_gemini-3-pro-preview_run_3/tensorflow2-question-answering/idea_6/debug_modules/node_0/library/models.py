import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import ModelConfig


class DualEncoderRanker(nn.Module):
    """
    Ranker model using a Dual-Encoder architecture with a shared Transformer backbone.
    Encodes text inputs into dense vectors via mean pooling.
    """

    def __init__(self):
        super(DualEncoderRanker, self).__init__()
        self.backbone = AutoModel.from_pretrained(ModelConfig.MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        """
        Encodes a batch of text sequences into dense embeddings.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, Seq_Len).
            attention_mask (torch.Tensor): Attention mask (Batch, Seq_Len).

        Returns:
            torch.Tensor: Pooled embeddings (Batch, Hidden_Size).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Mean Pooling
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings over valid tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum valid token counts (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings


class SimilarityProjectionReader(nn.Module):
    """
    Reader model that projects relevance signals onto context embeddings.
    Uses a similarity matrix between sequence tokens to augment representations
    before predicting span start/end logits.
    """

    def __init__(self):
        super(SimilarityProjectionReader, self).__init__()
        self.backbone = AutoModel.from_pretrained(ModelConfig.MODEL_NAME)
        self.hidden_size = ModelConfig.READER_HIDDEN_SIZE

        # Projection layer: Hidden Size + 1 (Similarity Score) -> 2 (Start/End Logits)
        self.qa_outputs = nn.Linear(self.hidden_size + 1, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """
        Computes start and end logits for answer spans.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, Seq_Len).
            attention_mask (torch.Tensor): Attention mask (Batch, Seq_Len).
            token_type_ids (torch.Tensor, optional): Segment IDs (0 for Query, 1 for Context).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Start logits and End logits.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (Batch, Seq_Len, Hidden)

        # 1. Compute Token-to-Token Similarity Matrix
        # (Batch, Seq, Hidden) @ (Batch, Hidden, Seq) -> (Batch, Seq, Seq)
        sim_matrix = torch.matmul(sequence_output, sequence_output.transpose(1, 2))

        # 2. Extract Relevance Signals (Max-Pooling along Question dimension)
        # We want to find how relevant each token in the sequence is to the Question.
        # Assuming standard BERT segmentation: 0 = Question, 1 = Context.
        # If token_type_ids are missing, we fallback to unmasked max-pooling (less ideal but functional).

        if token_type_ids is not None:
            # Create a mask for Question tokens (Batch, 1, Seq)
            # We want to keep columns corresponding to the Question (type 0)
            q_mask = (token_type_ids == 0).unsqueeze(1).type_as(sim_matrix)

            # Mask out non-question columns with a large negative value
            # (1.0 - q_mask) is 1 for Context tokens, so we suppress those columns
            sim_matrix = sim_matrix + (1.0 - q_mask) * -1e9

        # Max pooling along the last dimension (columns)
        # For each row (token), find its maximum similarity to any valid Question token
        # Shape: (Batch, Seq)
        relevance_scores, _ = torch.max(sim_matrix, dim=-1)

        # 3. Concatenate Relevance Scalar with Token Embeddings
        # Shape: (Batch, Seq, 1)
        relevance_scores = relevance_scores.unsqueeze(-1)

        # Shape: (Batch, Seq, Hidden + 1)
        combined_embeddings = torch.cat([sequence_output, relevance_scores], dim=-1)

        # 4. Project to Logits
        logits = self.qa_outputs(combined_embeddings)  # (Batch, Seq, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, Seq)
        end_logits = end_logits.squeeze(-1)  # (Batch, Seq)

        return start_logits, end_logits
