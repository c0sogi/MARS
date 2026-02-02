import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    """
    Configuration class for Cassava Leaf Disease Classification.
    """

    # General
    seed = 42
    debug = False
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    input_root = "./input"
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"
    output_dir = "./working/idea_3"

    # Model
    model_name = "vit_base_patch16_384"
    pretrained = True
    num_classes = 5
    img_size = 384

    # Training
    epochs = 10
    batch_size = 32  # Fits comfortably on A100 40GB with ViT-B/16 @ 384
    accum_iter = (
        1  # Gradient accumulation steps (Effective BS = batch_size * accum_iter)
    )
    lr = 2e-5  # Lower learning rate for fine-tuning pre-trained ViT
    min_lr = 1e-6
    weight_decay = 1e-2
    max_grad_norm = 1.0

    # Regularization & Augmentation
    label_smoothing = 0.1
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    mixup_prob = 0.5  # Probability of applying MixUp or CutMix

    # Inference
    tta_steps = 3  # Test Time Augmentation: Original + HFlip + VFlip

    # Logging
    print_freq = 100

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
