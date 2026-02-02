import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomBackbone(nn.Module):
    """
    A custom wrapper around Hugging Face Transformers to perform segment-aware pooling.

    This model extracts specific embeddings for the Question and Answer segments
    defined by the provided masks, as well as the CLS token and a difference vector.
    These features are designed to be used by Topology-Aware solvers.
    """

    def __init__(self, model_name):
        """
        Initialize the backbone model.

        Args:
            model_name (str): The name or path of the pre-trained transformer model.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, q_mask, a_mask):
        """
        Forward pass to extract segment-specific features.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, Seq_Len).
            attention_mask (torch.Tensor): Attention mask (Batch, Seq_Len).
            q_mask (torch.Tensor): Binary mask identifying Question tokens (Batch, Seq_Len).
            a_mask (torch.Tensor): Binary mask identifying Answer tokens (Batch, Seq_Len).

        Returns:
            dict: A dictionary containing:
                - 'h_cls': CLS token embedding (Batch, Hidden_Dim)
                - 'h_q': Mean pooled Question embedding (Batch, Hidden_Dim)
                - 'h_a': Mean pooled Answer embedding (Batch, Hidden_Dim)
                - 'h_diff': Absolute difference |h_q - h_a| (Batch, Hidden_Dim)
        """
        # Pass through the transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = (
            outputs.last_hidden_state
        )  # Shape: (Batch, Seq_Len, Hidden_Dim)

        # 1. Extract CLS embedding (Index 0)
        h_cls = last_hidden_state[:, 0, :]

        # 2. Prepare masks for broadcasting
        # Input masks are (Batch, Seq_Len), expand to (Batch, Seq_Len, 1)
        q_mask_expanded = q_mask.unsqueeze(-1)
        a_mask_expanded = a_mask.unsqueeze(-1)

        # 3. Compute Mean Pooling for Question (h_q)
        # Sum of hidden states where mask is active
        sum_q = torch.sum(last_hidden_state * q_mask_expanded, dim=1)
        # Count of active tokens (clamp to avoid division by zero)
        cnt_q = torch.sum(q_mask_expanded, dim=1).clamp(min=1e-9)
        h_q = sum_q / cnt_q

        # 4. Compute Mean Pooling for Answer (h_a)
        sum_a = torch.sum(last_hidden_state * a_mask_expanded, dim=1)
        cnt_a = torch.sum(a_mask_expanded, dim=1).clamp(min=1e-9)
        h_a = sum_a / cnt_a

        # 5. Compute Interaction Feature (h_diff)
        h_diff = torch.abs(h_q - h_a)

        return {"h_cls": h_cls, "h_q": h_q, "h_a": h_a, "h_diff": h_diff}
