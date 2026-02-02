import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DeepSupervisionLoss(nn.Module):
    """
    Implements the objective function for the Deeply-Supervised Wide-Stream Residual BiLSTM.

    Formula:
        L = MSE(y_main, y_gt) + lambda * Sum(MSE(y_layer_i, y_gt))

    Key Features:
    - Strictly uses MSE (L2) loss.
    - Applies masking by slicing predictions to the first 68 positions (Config.PRED_LEN).
    - Aggregates losses from the main output and all intermediate layers (Stem + Blocks).
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()
        self.lambda_weight = Config.DEEP_SUPERVISION_WEIGHT
        self.pred_len = Config.PRED_LEN

    def forward(self, main_pred, layer_preds, y_true):
        """
        Computes the weighted deep supervision loss.

        Args:
            main_pred (torch.Tensor): Final prediction from the scalar mixture.
                                      Shape: (Batch, Seq_Len, Channels)
            layer_preds (list[torch.Tensor]): List of predictions from intermediate layers.
                                              Each Shape: (Batch, Seq_Len, Channels)
            y_true (torch.Tensor): Ground truth targets.
                                   Shape: (Batch, Pred_Len, Channels)

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # Ensure inputs are float
        y_true = y_true.float()

        # 1. Slice Main Prediction to match scoring length (68)
        # Model output is full sequence length (107), target is (68)
        main_pred_scored = main_pred[:, : self.pred_len, :]

        # 2. Compute Main Loss (MSE)
        loss_main = F.mse_loss(main_pred_scored, y_true)

        # 3. Compute Deep Supervision Loss
        loss_deep = 0.0
        for pred_layer in layer_preds:
            # Slice intermediate prediction
            pred_layer_scored = pred_layer[:, : self.pred_len, :]

            # Compute MSE for this layer
            loss_deep += F.mse_loss(pred_layer_scored, y_true)

        # 4. Combine Losses
        total_loss = loss_main + (self.lambda_weight * loss_deep)

        return total_loss
