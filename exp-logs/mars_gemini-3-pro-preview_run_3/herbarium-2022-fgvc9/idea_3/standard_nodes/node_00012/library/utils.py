import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Mixup:
    """
    Applies Mixup and CutMix augmentation to a batch of images and labels.
    Handles target conversion to one-hot encoding and label smoothing.
    """

    def __init__(
        self,
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=Config.SWITCH_PROB,
        mode="batch",
        label_smoothing=Config.LABEL_SMOOTHING,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            mixup_alpha (float): Alpha parameter for Mixup Beta distribution.
            cutmix_alpha (float): Alpha parameter for CutMix Beta distribution.
            prob (float): Probability of applying either Mixup or CutMix.
            switch_prob (float): Probability of switching to CutMix given that augmentation is applied.
            mode (str): Mixing mode (currently only 'batch' is implemented).
            label_smoothing (float): Amount of label smoothing to apply (0.0 to 1.0).
            num_classes (int): Number of classes for one-hot encoding.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _rand_bbox(self, size, lam):
        """Generates a random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample center of the box
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x, target):
        """
        Args:
            x (Tensor): Input batch of images (N, C, H, W).
            target (Tensor): Input batch of class indices (N,).

        Returns:
            x (Tensor): Mixed images.
            target (Tensor): Mixed and smoothed soft targets (N, num_classes).
        """
        # Convert targets to one-hot and apply label smoothing
        # Target is expected to be on the same device as x eventually,
        # but F.one_hot creates a new tensor.
        with torch.no_grad():
            # Ensure target is on the correct device for one_hot
            target = target.to(x.device)
            target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()

            if self.label_smoothing > 0:
                target_one_hot = (
                    target_one_hot * (1 - self.label_smoothing)
                    + self.label_smoothing / self.num_classes
                )

            # Decide whether to apply augmentation
            if np.random.rand() > self.prob:
                return x, target_one_hot

            # Generate random permutation for mixing
            rand_index = torch.randperm(x.size(0)).to(x.device)

            # Decide between CutMix and Mixup
            use_cutmix = np.random.rand() < self.switch_prob

            if use_cutmix and self.cutmix_alpha > 0:
                # --- CutMix ---
                lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)

                # Adjust lambda to match the exact pixel count of the cropped area
                lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2])
                )

                # Apply CutMix to images
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

                # Mix targets
                target_mixed = (
                    lam * target_one_hot + (1 - lam) * target_one_hot[rand_index]
                )

            elif self.mixup_alpha > 0:
                # --- Mixup ---
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                # Apply Mixup to images
                x = lam * x + (1 - lam) * x[rand_index]

                # Mix targets
                target_mixed = (
                    lam * target_one_hot + (1 - lam) * target_one_hot[rand_index]
                )

            else:
                # Fallback if alphas are 0 but prob condition met
                target_mixed = target_one_hot

            return x, target_mixed


def get_label_mappings(csv_path=None):
    """
    Generates mappings between raw category_ids and contiguous indices.
    """
    if csv_path is None:
        csv_path = os.path.join(Config.METADATA_DIR, "train.csv")

    df = pd.read_csv(csv_path)
    unique_ids = sorted(df["category_id"].unique())
    label2id = {uid: idx for idx, uid in enumerate(unique_ids)}
    id2label = {idx: uid for idx, uid in enumerate(unique_ids)}
    return label2id, id2label
