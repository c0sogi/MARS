import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridLoss(nn.Module):
    """
    Hybrid Loss function combining Connectionist Temporal Classification (CTC) Loss
    and Cross-Entropy (CE) Loss for the Dual-Head architecture.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()

        # CTC Loss for the Encoder Head
        # zero_infinity=True helps prevent NaNs during early training if alignment is poor
        self.ctc_criterion = nn.CTCLoss(
            blank=Config.BLK_IDX, zero_infinity=True, reduction="mean"
        )

        # Cross Entropy Loss for the Decoder (Attention) Head
        # ignore_index handles the padding tokens in the target sequences
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=Config.PAD_IDX, reduction="mean"
        )

        self.ctc_weight = Config.CTC_WEIGHT

    def forward(self, ctc_logits, attn_logits, targets, target_lengths):
        """
        Args:
            ctc_logits (torch.Tensor): (Batch, Enc_Seq_Len, Vocab) - Output from Encoder CTC head
            attn_logits (torch.Tensor): (Batch, Dec_Seq_Len, Vocab) - Output from Decoder Attention head
            targets (torch.Tensor): (Batch, Max_Seq_Len) - Ground truth indices (including SOS/EOS/PAD)
            target_lengths (torch.Tensor): (Batch,) - Actual lengths of targets

        Returns:
            loss (torch.Tensor): Weighted sum of CTC and CE loss
            metrics (dict): Dictionary containing 'loss_ctc' and 'loss_attn' values
        """
        batch_size = ctc_logits.size(0)
        device = ctc_logits.device

        # ---------------------------------------------------------
        # 1. CTC Loss
        # ---------------------------------------------------------
        # CTC expects log_probs of shape (Input_Len, Batch, Class)
        # ctc_logits is (Batch, Input_Len, Class)
        log_probs = F.log_softmax(ctc_logits, dim=2).permute(1, 0, 2)

        # Input lengths for CTC:
        # We assume the full encoder output sequence is valid (padding regions will be learned as blanks)
        # Shape: (Batch,) filled with T_encoder
        enc_seq_len = ctc_logits.size(1)
        input_lengths = torch.full(
            size=(batch_size,), fill_value=enc_seq_len, dtype=torch.long
        ).to(device)

        # Targets for CTC:
        # We use the full padded targets. CTCLoss ignores indices beyond target_lengths.
        # Note: targets contain SOS/EOS. CTC will attempt to align them.
        # This is acceptable and often beneficial for end-to-end alignment.
        loss_ctc = self.ctc_criterion(log_probs, targets, input_lengths, target_lengths)

        # ---------------------------------------------------------
        # 2. Attention (Cross-Entropy) Loss
        # ---------------------------------------------------------
        loss_attn = torch.tensor(0.0, device=device)

        if attn_logits is not None:
            # Shift targets for teacher forcing:
            # Input to decoder was: [SOS, t1, t2, ..., tn, PAD]
            # We want to predict:   [t1, t2, ..., tn, EOS, PAD]

            # Predictions: Remove the last time step (corresponding to the last input)
            # Shape: (Batch, Seq_Len-1, Vocab)
            preds = attn_logits[:, :-1, :]

            # Targets: Remove the first time step (SOS)
            # Shape: (Batch, Seq_Len-1)
            # Note: targets typically include EOS, so shifting by 1 aligns predictions to next token
            ground_truth = targets[:, 1:]

            # Reshape for CrossEntropyLoss: (Batch, Vocab, Seq_Len) vs (Batch, Seq_Len)
            # Transpose preds to (Batch, Vocab, Seq_Len-1)
            loss_attn = self.ce_criterion(preds.transpose(1, 2), ground_truth)

        # ---------------------------------------------------------
        # 3. Aggregation
        # ---------------------------------------------------------
        total_loss = (self.ctc_weight * loss_ctc) + ((1 - self.ctc_weight) * loss_attn)

        metrics = {
            "loss_ctc": loss_ctc.item(),
            "loss_attn": loss_attn.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
