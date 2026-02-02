import torch
import torch.nn as nn
from library.config import Config


class MultiTaskLoss(nn.Module):
    """
    Computes the weighted multi-task loss for the Syntax-Aware Transformer.

    Components:
    1. Localization Loss (BCE): Did the model correctly identify the gap position?
       - Computed only on [GAP] tokens.
    2. Syntax Loss (CE): Did the model correctly predict the POS tag of the missing word?
       - Computed only at the ground-truth gap position.
    3. Identification Loss (CE): Did the model correctly predict the missing word ID?
       - Computed only at the ground-truth gap position.
    """

    def __init__(self):
        super().__init__()

        # Localization: Binary Classification (Missing vs Not Missing)
        # We use BCEWithLogitsLoss for numerical stability.
        self.loc_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        # Syntax: Multi-class Classification
        # ignore_index=-100 handles masking for non-target positions automatically
        self.syntax_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        # Identification: Multi-class Classification
        self.id_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        # Weights from Config
        self.lambda_loc = Config.LAMBDA_LOC
        self.lambda_syn = Config.LAMBDA_SYN
        self.lambda_id = Config.LAMBDA_ID

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Output from the model containing:
                - 'loc_logits': (batch_size, seq_len)
                - 'syntax_logits': (batch_size, seq_len, num_pos_tags)
                - 'word_logits': (batch_size, seq_len, vocab_size)
            batch (dict): Batch data containing:
                - 'gap_mask': (batch_size, seq_len) - 1 for GAP tokens, 0 otherwise
                - 'loc_targets': (batch_size, seq_len) - 1.0 at target gap, 0.0 otherwise
                - 'syntax_targets': (batch_size, seq_len) - POS ID at target, -100 otherwise
                - 'word_targets': (batch_size, seq_len) - Word ID at target, -100 otherwise

        Returns:
            dict: Dictionary containing 'loss' (total weighted loss) and individual components.
        """
        # Unpack outputs
        loc_logits = outputs["loc_logits"]  # (B, S)
        syntax_logits = outputs["syntax_logits"]  # (B, S, Num_Tags)
        word_logits = outputs["word_logits"]  # (B, S, Vocab_Size)

        # Unpack targets
        gap_mask = batch["gap_mask"]  # (B, S)
        loc_targets = batch["loc_targets"]  # (B, S)
        syntax_targets = batch["syntax_targets"]  # (B, S)
        word_targets = batch["word_targets"]  # (B, S)

        # ----------------------------------------------------------------------
        # 1. Localization Loss
        # ----------------------------------------------------------------------
        # We only care about predictions at GAP tokens.
        # Filter logits and targets using the gap_mask.

        # Flatten tensors
        loc_logits_flat = loc_logits.view(-1)
        loc_targets_flat = loc_targets.view(-1)
        gap_mask_flat = gap_mask.view(-1).bool()

        # Select only gap positions
        valid_loc_logits = loc_logits_flat[gap_mask_flat]
        valid_loc_targets = loc_targets_flat[gap_mask_flat]

        if valid_loc_logits.numel() > 0:
            loc_loss = self.loc_loss_fn(valid_loc_logits, valid_loc_targets)
        else:
            loc_loss = torch.tensor(0.0, device=loc_logits.device)

        # ----------------------------------------------------------------------
        # 2. Syntax Loss
        # ----------------------------------------------------------------------
        # CrossEntropyLoss expects (N, C) or (N, C, d1...)
        # We permute logits to (B, C, S) for compatibility with (B, S) targets

        syntax_logits_permuted = syntax_logits.permute(0, 2, 1)  # (B, Num_Tags, S)
        syn_loss = self.syntax_loss_fn(syntax_logits_permuted, syntax_targets)

        # ----------------------------------------------------------------------
        # 3. Identification Loss
        # ----------------------------------------------------------------------
        word_logits_permuted = word_logits.permute(0, 2, 1)  # (B, Vocab_Size, S)
        id_loss = self.id_loss_fn(word_logits_permuted, word_targets)

        # ----------------------------------------------------------------------
        # Total Loss
        # ----------------------------------------------------------------------
        total_loss = (
            (self.lambda_loc * loc_loss)
            + (self.lambda_syn * syn_loss)
            + (self.lambda_id * id_loss)
        )

        return {
            "loss": total_loss,
            "loc_loss": loc_loss.detach(),
            "syn_loss": syn_loss.detach(),
            "id_loss": id_loss.detach(),
        }
