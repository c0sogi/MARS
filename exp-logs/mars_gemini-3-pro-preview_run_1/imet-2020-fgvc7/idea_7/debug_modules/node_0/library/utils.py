import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=Config.SEED):
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


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to improve robustness and generalization.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=None):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.device = device
        if self.device:
            self.model.to(self.device)

    def update(self, model):
        """
        Update the EMA model parameters.
        Args:
            model: The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for name, param in self.model.named_parameters():
                if name in msd:
                    new_param = msd[name].to(param.device)
                    param.data.mul_(self.decay).add_(new_param, alpha=1 - self.decay)

            # Update buffers (e.g., BatchNorm running mean/var) by direct copy
            for name, buf in self.model.named_buffers():
                if name in msd:
                    new_buf = msd[name].to(buf.device)
                    buf.data.copy_(new_buf)

    @property
    def module(self):
        """Returns the underlying EMA model."""
        return self.model


class Mixup:
    """
    Implements Mixup and CutMix data augmentation.
    """

    def __init__(
        self,
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def rand_bbox(self, size, lam):
        """Generates a random bounding box for CutMix."""
        H, W = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        # Random center
        cy = np.random.randint(H)
        cx = np.random.randint(W)

        # Bounding box coordinates
        bbh1 = np.clip(cy - cut_h // 2, 0, H)
        bbh2 = np.clip(cy + cut_h // 2, 0, H)
        bbw1 = np.clip(cx - cut_w // 2, 0, W)
        bbw2 = np.clip(cx + cut_w // 2, 0, W)

        return bbh1, bbw1, bbh2, bbw2

    def __call__(self, x, y):
        """
        Applies Mixup or CutMix to the batch.
        Args:
            x: Input images (B, C, H, W)
            y: Input labels (B, NumClasses)
        Returns:
            mixed_x, mixed_y
        """
        if np.random.rand() > self.prob:
            return x, y

        batch_size = x.size(0)
        indices = torch.randperm(batch_size, device=x.device)

        # Decide between Mixup and CutMix
        if np.random.rand() < self.switch_prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = x * lam + x[indices] * (1 - lam)
            y = y * lam + y[indices] * (1 - lam)
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbh1, bbw1, bbh2, bbw2 = self.rand_bbox(x.size(), lam)

            # Adjust lambda based on the exact pixel area removed
            lam = 1 - ((bbh2 - bbh1) * (bbw2 - bbw1) / (x.size()[-1] * x.size()[-2]))

            x[:, :, bbh1:bbh2, bbw1:bbw2] = x[indices, :, bbh1:bbh2, bbw1:bbw2]
            y = y * lam + y[indices] * (1 - lam)

        return x, y


def calculate_f1(y_true, y_pred):
    """
    Calculates Micro F1 score.
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
    """
    return f1_score(y_true, y_pred, average="micro")


def optimize_threshold(y_true, y_pred_probs):
    """
    Performs grid search to find the optimal probability threshold for Micro F1.
    Args:
        y_true: Ground truth labels (N, C)
        y_pred_probs: Predicted probabilities (N, C)
    Returns:
        best_thr: Optimal threshold value
        best_score: Best Micro F1 score achieved
    """
    best_thr = 0.5
    best_score = 0.0

    # Define search range
    start = Config.THRESHOLD_START
    end = Config.THRESHOLD_END
    step = Config.THRESHOLD_STEP

    # Generate thresholds (using round to avoid floating point drift)
    thresholds = [round(x, 2) for x in np.arange(start, end + step, step)]

    for thr in thresholds:
        # Binarize predictions based on current threshold
        y_pred_bin = (y_pred_probs > thr).astype(int)

        # Calculate metric
        score = f1_score(y_true, y_pred_bin, average="micro")

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score
