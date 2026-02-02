import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetKLLoss(nn.Module):
    """
    Computes the KL Divergence loss between predicted logits and soft targets.
    Used for the main task loss with Gaussian-smoothed targets.
    """

    def __init__(self, reduction="batchmean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output.
                             'batchmean' is recommended for KLDivLoss to align with mathematical definition.
        """
        super(SoftTargetKLLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len).
            targets (torch.Tensor): Soft target probabilities of shape (batch_size, seq_len).
                                    These should sum to 1 along the seq_len dimension.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Compute log probabilities from logits (required for KLDivLoss input)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute KL Divergence
        # nn.KLDivLoss expects:
        #   input: log-probabilities
        #   target: probabilities
        loss = self.kl_loss(log_probs, targets)

        return loss


class RDropLoss(nn.Module):
    """
    Computes the R-Drop consistency regularization loss.
    Calculates the bidirectional KL divergence between two sets of logits
    generated from the same input with different dropout masks.
    """

    def __init__(self, reduction="batchmean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output.
        """
        super(RDropLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits1, logits2):
        """
        Args:
            logits1 (torch.Tensor): Logits from the first forward pass. Shape (batch_size, seq_len).
            logits2 (torch.Tensor): Logits from the second forward pass. Shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The calculated consistency loss (scalar).
        """
        # Compute log probabilities for both outputs
        p_log = F.log_softmax(logits1, dim=-1)
        q_log = F.log_softmax(logits2, dim=-1)

        # Compute probabilities (targets) for both outputs
        p_prob = F.softmax(logits1, dim=-1)
        q_prob = F.softmax(logits2, dim=-1)

        # KL(P || Q) = sum(P * (log P - log Q))
        # input: q_log (log Q), target: p_prob (P)
        kl_p_q = self.kl_loss(q_log, p_prob)

        # KL(Q || P) = sum(Q * (log Q - log P))
        # input: p_log (log P), target: q_prob (Q)
        kl_q_p = self.kl_loss(p_log, q_prob)

        # Average the bidirectional KL divergence
        loss = 0.5 * (kl_p_q + kl_q_p)

        return loss
