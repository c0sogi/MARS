import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import AverageMeter
from library.data import mixup_data, mixup_criterion


class Engine:
    """
    Engine class encapsulates the training, validation, and inference logic.
    """

    @staticmethod
    def train_one_epoch(model, loader, optimizer, device, epoch):
        """
        Trains the model for one epoch using Mixup and Auxiliary Loss.
        """
        model.train()
        losses = AverageMeter()

        # Binary Cross Entropy with Logits
        criterion = nn.BCEWithLogitsLoss()

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            # Ensure targets are (N, 1) floats
            targets = targets.float().to(device).view(-1, 1)

            # Apply Mixup
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.MIXUP_ALPHA, device
            )

            optimizer.zero_grad()

            # Forward pass: returns (main_output, aux_output)
            outputs, aux_outputs = model(images)

            # Calculate combined loss
            loss_main = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss_aux = mixup_criterion(
                criterion, aux_outputs, targets_a, targets_b, lam
            )

            loss = loss_main + Config.AUX_WEIGHT * loss_aux

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

        print(f"Epoch {epoch}: Train Loss {losses.avg}")
        return losses.avg

    @staticmethod
    def validate(model, loader, device):
        """
        Evaluates the model on the validation set.
        Returns average loss and AUC.
        """
        model.eval()
        losses = AverageMeter()
        criterion = nn.BCEWithLogitsLoss()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.float().to(device).view(-1, 1)

                # Forward pass (ignore aux head for validation)
                outputs, _ = model(images)

                loss = criterion(outputs, targets)
                losses.update(loss.item(), images.size(0))

                # Apply sigmoid for probabilities
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            # Handle edge cases (e.g., only one class in batch)
            auc = 0.5

        print(f"Validation Loss: {losses.avg}")
        print(f"Validation AUC: {auc}")

        return losses.avg, auc

    @staticmethod
    def update_swa(swa_model, model):
        """
        Updates the SWA model parameters.
        """
        swa_model.update_parameters(model)

    @staticmethod
    def predict_tta_raw(model, loader, device):
        """
        Performs inference with Test Time Augmentation (4 views).
        Returns raw probabilities for each view and the corresponding IDs/Targets.

        Views:
        1. Original
        2. Horizontal Flip
        3. Vertical Flip
        4. 180 Degree Rotation

        Returns:
            probs (np.ndarray): Shape (N, 4) - Probabilities for each view.
            ids (np.ndarray): Shape (N,) - Corresponding IDs or Targets.
        """
        model.eval()

        all_probs = []
        all_ids = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)

                # Collect IDs/Targets
                if isinstance(targets, torch.Tensor):
                    batch_ids = targets.cpu().numpy()
                else:
                    batch_ids = np.array(targets)
                all_ids.extend(batch_ids)

                # 1. Original
                out1, _ = model(images)
                prob1 = torch.sigmoid(out1)

                # 2. Horizontal Flip (Flip Width - dim 3)
                img_h = torch.flip(images, dims=[3])
                out2, _ = model(img_h)
                prob2 = torch.sigmoid(out2)

                # 3. Vertical Flip (Flip Height - dim 2)
                img_v = torch.flip(images, dims=[2])
                out3, _ = model(img_v)
                prob3 = torch.sigmoid(out3)

                # 4. Rotate 180 (Flip H + Flip W - dims 2, 3)
                img_r = torch.flip(images, dims=[2, 3])
                out4, _ = model(img_r)
                prob4 = torch.sigmoid(out4)

                # Stack probabilities: (Batch, 4)
                batch_probs = torch.cat([prob1, prob2, prob3, prob4], dim=1)
                all_probs.append(batch_probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0), np.array(all_ids)
