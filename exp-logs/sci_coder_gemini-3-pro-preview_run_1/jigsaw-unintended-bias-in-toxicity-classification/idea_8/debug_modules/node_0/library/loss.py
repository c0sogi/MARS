import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class CompositeLoss(_Loss):
    """
    Composite Loss function combining:
    1. Weighted Binary Cross Entropy (Pointwise)
    2. Pairwise Margin Ranking Loss (Ranking)
    3. Auxiliary Multi-Task Loss (Regularization)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.aux_bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.margin_rank = nn.MarginRankingLoss(margin=0.5, reduction="mean")

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Output from ToxicityModel containing 'toxicity', 'identity', 'attack'.
            batch (dict): Batch dictionary from ToxicityDataset containing targets and weights.

        Returns:
            torch.Tensor: The total combined loss.
            dict: Dictionary of individual loss components for logging.
        """
        # Unpack inputs
        toxicity_logits = outputs["toxicity"].view(-1)
        identity_logits = outputs["identity"]
        attack_logits = outputs["attack"].view(-1)

        targets = batch["target"]
        aux_targets = batch["aux_targets"]
        attack_target = batch["attack_target"]
        sample_weights = batch["sample_weight"]

        # 1. Weighted Pointwise Loss (Primary Objective)
        # We apply the sample weights (which upweight bias traps) to the BCE loss
        bce_loss_per_sample = self.bce(toxicity_logits, targets)
        weighted_bce_loss = (bce_loss_per_sample * sample_weights).mean()

        # 2. Auxiliary Loss (Semantic Disentanglement)
        # Standard BCE for identity attributes and identity attack subtype
        loss_identity = self.aux_bce(identity_logits, aux_targets)
        loss_attack = self.aux_bce(attack_logits, attack_target)
        aux_loss = loss_identity + loss_attack

        # 3. Pairwise Margin Ranking Loss (Bias Mitigation)
        # We want: Score(Toxic + NoIdentity) > Score(NonToxic + Identity)

        # Derive masks based on targets
        # identity_mentioned is true if any identity column >= 0.5
        identity_mentioned = (aux_targets >= 0.5).any(dim=1)

        # Positive Class (Toxic)
        is_toxic = targets >= 0.5

        # Define the specific subgroups for ranking
        # Group A: Toxic AND No Identity Mention (Background Positive)
        mask_pos = is_toxic & (~identity_mentioned)

        # Group B: Non-Toxic AND Identity Mention (Subgroup Negative / Bias Trap)
        mask_neg = (~is_toxic) & identity_mentioned

        # Extract logits for these groups
        preds_pos = toxicity_logits[mask_pos]
        preds_neg = toxicity_logits[mask_neg]

        ranking_loss = torch.tensor(0.0, device=self.config.device)

        # We can only compute ranking loss if we have at least one example in both groups
        if preds_pos.size(0) > 0 and preds_neg.size(0) > 0:
            # Create all pairs (broadcast)
            # preds_pos: (N,) -> (N, 1)
            # preds_neg: (M,) -> (1, M)
            # We want maximize margin: pos - neg > margin
            # MarginRankingLoss expects inputs x1, x2, y. Loss is max(0, -y * (x1 - x2) + margin)
            # If y=1, Loss is max(0, margin - (x1 - x2))

            # Expand to compare every positive against every negative
            n_pos = preds_pos.size(0)
            n_neg = preds_neg.size(0)

            x1 = preds_pos.unsqueeze(1).expand(n_pos, n_neg).reshape(-1)
            x2 = preds_neg.unsqueeze(0).expand(n_pos, n_neg).reshape(-1)

            # Target is 1 because we want x1 > x2
            target_rank = torch.ones_like(x1)

            ranking_loss = self.margin_rank(x1, x2, target_rank)

        # Combine Losses
        total_loss = (
            weighted_bce_loss
            + (self.config.lambda_rank * ranking_loss)
            + (self.config.lambda_aux * aux_loss)
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_bce": weighted_bce_loss.item(),
            "loss_rank": ranking_loss.item(),
            "loss_aux": aux_loss.item(),
        }

        return total_loss, loss_dict


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights in the direction of the gradient ascent to find flat minima.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps):
        """
        Args:
            model (nn.Module): The model to perturb.
            optimizer (optim.Optimizer): The optimizer used for training.
            adv_lr (float): Learning rate for the adversary (perturbation magnitude scaler).
            adv_eps (float): Epsilon limit for the perturbation norm.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the weights.
        Should be called after loss.backward() so that gradients are populated.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Only perturb parameters that require gradients and have gradients
            if param.requires_grad and param.grad is not None:
                # Save original weights
                self.backup[name] = param.data.clone()

                # Calculate perturbation direction
                grad = param.grad
                norm = torch.norm(grad)

                if norm != 0 and not torch.isnan(norm):
                    # Direction = grad / norm
                    # Perturbation = adv_lr * direction
                    r_at = self.adv_lr * grad / (norm + e)

                    # Add perturbation to weights
                    param.data.add_(r_at)
                    self.backup_eps[name] = r_at

    def restore(self):
        """
        Restores the original model weights.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        self.backup_eps = {}
