import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        # NLL loss part: -log(p(y))
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        # Smooth part: -mean(log(p(k))) over all k
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Mixup:
    """
    Handles Mixup data augmentation logic.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Performs mixup on the input batch x and targets y.
        Returns: mixed_x, y_a, y_b, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        if torch.cuda.is_available():
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def criterion(criterion, pred, y_a, y_b, lam):
        """
        Computes the mixup loss given the criterion, predictions, and mixed targets.
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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


def get_score(y_true, y_pred):
    """
    Calculates the categorization accuracy.
    Args:
        y_true: list or array of ground truth labels
        y_pred: list or array of predicted labels
    Returns:
        float: Accuracy score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_true == y_pred).mean()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Saves the model checkpoint.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, optimizer=None, filename="model_best.pth.tar"):
    """
    Loads a checkpoint into the model and optimizer.
    Returns: best_acc, start_epoch
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        print(f"=> loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        best_acc = checkpoint.get("best_acc", 0)
        epoch = checkpoint.get("epoch", 0)
        print(f"=> loaded checkpoint '{filepath}' (epoch {epoch}, acc {best_acc})")
        return best_acc, epoch
    else:
        print(f"=> no checkpoint found at '{filepath}'")
        return 0, 0
