import torch
import torch.nn as nn


class HierarchicalCompoundLoss(nn.Module):
    """
    Hierarchical Compound Loss for Cervical Spine Fracture Detection.

    This loss function implements the strategy of combining vertebral-level supervision
    with patient-level supervision. It computes the sum of:
    1. The mean Binary Cross Entropy (BCE) loss across the 7 vertebral sub-types.
    2. The BCE loss for the 'patient_overall' prediction.

    By summing the mean of the vertebral losses with the patient loss, the patient-level
    outcome is implicitly weighted more heavily than any single vertebral prediction,
    encouraging the model to prioritize global consistency.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output of BCEWithLogitsLoss.
                             Default: 'mean'.
        """
        super(HierarchicalCompoundLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, outputs, targets):
        """
        Computes the hierarchical compound loss.

        Args:
            outputs (dict): A dictionary containing the model predictions.
                - 'vertebrae_logits': Tensor of shape (Batch, 7) containing logits for C1-C7.
                - 'patient_logit': Tensor of shape (Batch, 1) containing the logit for patient_overall.
            targets (dict): A dictionary containing the ground truth labels.
                - 'vertebrae': Tensor of shape (Batch, 7) containing binary labels (0 or 1).
                - 'patient_overall': Tensor of shape (Batch) or (Batch, 1) containing binary labels.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # 1. Extract Model Predictions
        pred_vert = outputs["vertebrae_logits"]
        pred_patient = outputs["patient_logit"]

        # 2. Extract Ground Truth Targets
        target_vert = targets["vertebrae"]
        target_patient = targets["patient_overall"]

        # 3. Shape Standardization
        # Ensure patient target is (Batch, 1) to match pred_patient
        if target_patient.dim() == 1:
            target_patient = target_patient.view(-1, 1)

        # Ensure targets are on the correct device (same as predictions)
        if target_vert.device != pred_vert.device:
            target_vert = target_vert.to(pred_vert.device)
        if target_patient.device != pred_patient.device:
            target_patient = target_patient.to(pred_patient.device)

        # 4. Compute Losses
        # Vertebral Loss: Average BCE across all 7 vertebrae (and batch)
        # This treats each vertebra prediction equally.
        loss_vert = self.bce(pred_vert, target_vert)

        # Patient Loss: BCE for the single global outcome
        loss_patient = self.bce(pred_patient, target_patient)

        # 5. Combine Losses (Implicit Weighting)
        # Adding the patient loss (scalar) to the mean of vertebral losses (scalar)
        # effectively gives the patient outcome a higher relative weight compared to
        # an individual vertebra's contribution to the mean.
        total_loss = loss_vert + loss_patient

        return total_loss
