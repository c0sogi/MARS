import os
import random
import numpy as np
import torch
from library.config import LabelConfig


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class LabelMapper:
    """
    Handles the mapping between the 31 fine-grained training classes and the
    12 target submission classes.
    """

    def __init__(self):
        # Load the vocabulary from config
        self.fine_grained_labels = LabelConfig.fine_grained_labels
        self.num_classes = len(self.fine_grained_labels)

        # Bidirectional mapping for model training (String <-> Index)
        self.label2idx = {
            label: idx for idx, label in enumerate(self.fine_grained_labels)
        }
        self.idx2label = {
            idx: label for idx, label in enumerate(self.fine_grained_labels)
        }

        # Mapping for submission (Fine-grained -> Target/Unknown)
        self.submission_map = LabelConfig.get_label_map()

    def encode(self, labels):
        """
        Converts a list of fine-grained string labels to a tensor of indices.
        Args:
            labels (list[str]): List of label strings.
        Returns:
            torch.Tensor: Tensor of integer indices.
        """
        indices = [self.label2idx[label] for label in labels]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """
        Converts a tensor or list of indices to a list of fine-grained string labels.
        Args:
            indices (torch.Tensor or list): Integer indices.
        Returns:
            list[str]: List of label strings.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        return [self.idx2label[idx] for idx in indices]

    def map_to_submission(self, fine_grained_label):
        """
        Maps a single fine-grained label to the required submission format.
        Logic:
            - Target commands -> Target commands
            - Silence -> Silence
            - Auxiliary words -> Unknown
        """
        return self.submission_map.get(fine_grained_label, "unknown")

    def map_indices_to_submission(self, indices):
        """
        Helper to convert model output indices directly to submission labels.
        """
        fine_labels = self.decode(indices)
        return [self.map_to_submission(l) for l in fine_labels]


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Args:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup interpolation coefficient.
        device (str): Device to perform permutation on.
    Returns:
        mixed_x: Mixed input tensor.
        y_a: Labels of the first sample.
        y_b: Labels of the second sample.
        lam: Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs.
    Args:
        criterion: Loss function.
        pred: Model predictions.
        y_a: First set of targets.
        y_b: Second set of targets.
        lam: Lambda mixing coefficient.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
