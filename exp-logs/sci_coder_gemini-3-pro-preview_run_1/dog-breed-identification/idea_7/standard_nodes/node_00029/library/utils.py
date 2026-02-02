import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_log_loss(y_true, y_pred):
    """
    Calculates the Multi Class Log Loss.

    Args:
        y_true: Ground truth labels. Can be a 1D tensor/array of indices or 2D one-hot encoded.
        y_pred: Predicted probabilities. 2D tensor/array (N_samples, N_classes).

    Returns:
        float: The log loss value.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Define all possible labels to ensure log_loss works even if a batch misses some classes
    # Assuming classes are indexed 0 to 119 based on the config
    all_labels = list(range(120))

    # sklearn log_loss handles both label indices and one-hot targets
    return log_loss(y_true, y_pred, labels=all_labels)


class MixupCutmix:
    """
    Implements Mixup and CutMix augmentation for regularization.
    Applies mixing to both images and labels (creating soft targets).
    """

    def __init__(
        self, mixup_alpha=1.0, cutmix_alpha=1.0, mix_prob=0.5, num_classes=120
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.num_classes = num_classes

    def __call__(self, batch_images, batch_labels):
        """
        Args:
            batch_images (Tensor): Batch of images (B, C, H, W)
            batch_labels (Tensor): Batch of labels (B) or (B, num_classes)

        Returns:
            mixed_images, mixed_labels
        """
        # Decide whether to apply any mixing
        if np.random.rand() > self.mix_prob:
            # If input labels are indices, convert to one-hot to maintain consistency
            # for the loss function which expects soft targets when mixing is enabled in the pipeline
            if len(batch_labels.shape) == 1:
                batch_labels = F.one_hot(
                    batch_labels, num_classes=self.num_classes
                ).float()
            return batch_images, batch_labels

        # Decide between Mixup and CutMix (50/50 split)
        use_cutmix = (np.random.rand() > 0.5) and (self.cutmix_alpha > 0)

        if use_cutmix:
            return self._cutmix(batch_images, batch_labels)
        else:
            return self._mixup(batch_images, batch_labels)

    def _mixup(self, images, labels):
        batch_size = images.size(0)
        device = images.device

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Shuffle batch
        index = torch.randperm(batch_size, device=device)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index, :]

        # Mix labels
        if len(labels.shape) == 1:
            labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            labels_one_hot = labels

        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index, :]

        return mixed_images, mixed_labels

    def _cutmix(self, images, labels):
        batch_size, _, h, w = images.shape
        device = images.device

        # Sample lambda
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        # Calculate bounding box dimensions
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Sample center of the box
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Calculate coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Shuffle batch
        index = torch.randperm(batch_size, device=device)

        # Adjust lambda to the exact area ratio of the crop
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        # Create mixed images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            index, :, bby1:bby2, bbx1:bbx2
        ]

        # Mix labels
        if len(labels.shape) == 1:
            labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            labels_one_hot = labels

        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index, :]

        return mixed_images, mixed_labels
