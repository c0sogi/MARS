import torch
import torch.nn as nn
from library.config import Config


class DecoupledMultiTaskLoss(nn.Module):
    """
    Implements the Decoupled Multi-Task Loss function.

    Equation:
    L_total = L_Main + lambda * (L_Aux_Rust + L_Aux_Scab + L_Aux_Healthy)

    Components:
    - Main Task: CrossEntropyLoss with Inverse Frequency Class Weights.
      Prioritizes the minority "Multiple Diseases" class.
    - Aux Tasks: BCEWithLogitsLoss.
      Decouples the learning of constituent features (Rust, Scab, Healthy).
    """

    def __init__(self, class_weights=None, device=None):
        """
        Args:
            class_weights (np.ndarray or torch.Tensor, optional): Weights for the main classes.
            device (torch.device, optional): Device to move weights to.
        """
        super(DecoupledMultiTaskLoss, self).__init__()

        # Initialize Main Criterion (CrossEntropy)
        # We handle class weights conversion to tensor/device here
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)

            if device is not None:
                class_weights = class_weights.to(device)

            self.main_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.main_criterion = nn.CrossEntropyLoss()

        # Initialize Auxiliary Criterion (BCE)
        self.aux_criterion = nn.BCEWithLogitsLoss()

        # Load Lambda from Config
        self.lambda_val = Config.LAMBDA

    def forward(self, outputs, targets):
        """
        Computes the weighted multi-task loss.

        Args:
            outputs (dict): Dictionary containing model logits.
                Keys: 'main', 'aux_rust', 'aux_scab', 'aux_healthy'
            targets (dict): Dictionary containing ground truth labels.
                Keys: 'main' (one-hot), 'aux_rust', 'aux_scab', 'aux_healthy' (binary)

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss (torch.Tensor): Scalar loss for backprop.
                - loss_dict (dict): Dictionary of individual loss components for logging.
        """
        # 1. Main Task Loss
        # Dataset returns one-hot vectors for 'main', but CrossEntropyLoss expects class indices.
        # We convert using argmax.
        main_logits = outputs["main"]
        main_targets_one_hot = targets["main"]
        main_targets_indices = torch.argmax(main_targets_one_hot, dim=1)

        loss_main = self.main_criterion(main_logits, main_targets_indices)

        # 2. Auxiliary Task Losses
        # Inputs are logits (model output) and binary float targets (dataset output)
        loss_aux_rust = self.aux_criterion(outputs["aux_rust"], targets["aux_rust"])
        loss_aux_scab = self.aux_criterion(outputs["aux_scab"], targets["aux_scab"])
        loss_aux_healthy = self.aux_criterion(
            outputs["aux_healthy"], targets["aux_healthy"]
        )

        # 3. Aggregation
        aux_loss_sum = loss_aux_rust + loss_aux_scab + loss_aux_healthy
        total_loss = loss_main + (self.lambda_val * aux_loss_sum)

        # 4. Logging Dictionary
        # Detach items for logging to avoid graph retention issues in training loops if stored
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_main": loss_main.item(),
            "loss_aux_rust": loss_aux_rust.item(),
            "loss_aux_scab": loss_aux_scab.item(),
            "loss_aux_healthy": loss_aux_healthy.item(),
        }

        return total_loss, loss_dict
