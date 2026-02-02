import torch
import torch.nn as nn
from library.config import Config


class MultiObjectiveGapLoss(nn.Module):
    """
    Computes the multi-objective loss for the Global-Localization Interleaved Transformer.

    Components:
    1. Localization Loss (CrossEntropy): Predicts which [GAP] token corresponds to the missing word.
    2. Identification Loss (CrossEntropy): Predicts the missing word ID at the correct gap location.
    3. Latent Alignment Loss (CosineEmbedding): Aligns the hidden state of the correct gap with the
       target word's embedding.
    """

    def __init__(
        self, lambda_align: float = Config.LAMBDA_ALIGN, ignore_index: int = -100
    ):
        """
        Args:
            lambda_align (float): Weighting factor for the alignment loss component.
            ignore_index (int): Index to ignore in loss calculation (used for padding/truncation).
        """
        super().__init__()
        self.lambda_align = lambda_align
        self.ignore_index = ignore_index

        # 1. Localization Loss
        # Input: (Batch, Seq_Len), Target: (Batch) containing index of correct gap
        self.loc_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # 2. Identification Loss
        # Input: (Batch, Vocab), Target: (Batch) containing word ID
        self.id_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # 3. Alignment Loss
        # Input: (Batch, Hidden), (Batch, Hidden), Target: 1 (similar)
        self.align_loss_fn = nn.CosineEmbeddingLoss()

    def forward(
        self,
        loc_logits: torch.Tensor,
        id_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        target_loc: torch.Tensor,
        target_id: torch.Tensor,
        embedding_layer: nn.Embedding,
    ) -> dict:
        """
        Computes the total weighted loss.

        Args:
            loc_logits (Tensor): Shape (Batch, Seq_Len). Logits for gap localization.
            id_logits (Tensor): Shape (Batch, Seq_Len, Vocab_Size). Logits for word identification.
            hidden_states (Tensor): Shape (Batch, Seq_Len, Embed_Dim). Raw hidden states.
            target_loc (Tensor): Shape (Batch). Index of the correct gap per sample.
            target_id (Tensor): Shape (Batch). ID of the missing word per sample.
            embedding_layer (nn.Embedding): The model's embedding layer to retrieve target vectors.

        Returns:
            dict: Dictionary containing 'loss' (total), 'loc_loss', 'id_loss', and 'align_loss'.
        """
        # 1. Localization Loss
        # Directly compares the distribution over the sequence to the target index
        loc_loss = self.loc_loss_fn(loc_logits, target_loc)

        # Filter out ignored/truncated targets for ID and Alignment losses
        # target_loc contains -100 for invalid samples
        valid_mask = target_loc != self.ignore_index

        # If no valid samples in batch (edge case), return only loc_loss (or zero)
        if not valid_mask.any():
            return {
                "loss": loc_loss,
                "loc_loss": loc_loss,
                "id_loss": torch.tensor(0.0, device=loc_logits.device),
                "align_loss": torch.tensor(0.0, device=loc_logits.device),
            }

        # Select valid indices
        valid_batch_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_target_locs = target_loc[valid_mask]
        valid_target_ids = target_id[valid_mask]

        # 2. Identification Loss
        # Extract logits only at the correct gap positions
        # id_logits: (Batch, Seq_Len, Vocab) -> select (Valid_Batch, Correct_Loc, Vocab)
        selected_id_logits = id_logits[valid_batch_indices, valid_target_locs]
        id_loss = self.id_loss_fn(selected_id_logits, valid_target_ids)

        # 3. Alignment Loss
        # Extract hidden states at correct gap positions
        selected_hidden = hidden_states[valid_batch_indices, valid_target_locs]

        # Get ground truth embeddings for the missing words
        # Detach embeddings to prevent updating the embedding layer via the alignment objective directly?
        # Strategy: We generally want the hidden state to move towards the embedding.
        # Updating embeddings via this loss might be unstable, but standard practice varies.
        # Here we treat embeddings as the fixed target for the hidden state in this specific loss term.
        target_embeddings = embedding_layer(valid_target_ids)

        # Target label 1 indicates we want vectors to be similar
        target_similarity = torch.ones(
            len(valid_batch_indices), device=loc_logits.device
        )

        align_loss = self.align_loss_fn(
            selected_hidden, target_embeddings, target_similarity
        )

        # Total Loss
        total_loss = loc_loss + id_loss + (self.lambda_align * align_loss)

        return {
            "loss": total_loss,
            "loc_loss": loc_loss,
            "id_loss": id_loss,
            "align_loss": align_loss,
        }
