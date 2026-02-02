import os
import sys
import shutil
import torch
import logging
from library.config import CHECKPOINT_DIR


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        # Returns a string representation, but full precision values
        # should be accessed via .avg directly in the training loop.
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Saves the model checkpoint.
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(CHECKPOINT_DIR, "model_best.pth.tar"))


def load_checkpoint(model, optimizer=None, filename="model_best.pth.tar"):
    """
    Loads a checkpoint from the CHECKPOINT_DIR.
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        print(f"=> loading checkpoint '{filepath}'")
        # Load to CPU first to avoid potential OOM if GPU is already occupied
        checkpoint = torch.load(filepath, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]

        # Handle potential DataParallel prefix in state_dict
        state_dict = checkpoint["state_dict"]
        if list(state_dict.keys())[0].startswith("module.") and not hasattr(
            model, "module"
        ):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> loaded checkpoint '{filepath}' (epoch {start_epoch})")
        return start_epoch, best_acc1
    else:
        print(f"=> no checkpoint found at '{filepath}'")
        return 0, 0.0


def get_logger(log_file):
    """
    Creates a logger that writes to a file and stdout.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
