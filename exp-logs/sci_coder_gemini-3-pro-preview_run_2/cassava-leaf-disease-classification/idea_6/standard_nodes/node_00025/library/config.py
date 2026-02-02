import os
import torch


class CFG:
    # General Config
    debug = False
    seed = 42
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    output_dir = "./working/idea_6"

    # Data Config
    image_size = 384
    num_classes = 5
    n_fold = 5

    # Model Config
    # Using the 384 resolution pretrained weights for better initialization
    model_name = "convnext_small.fb_in22k_ft_in1k_384"
    drop_path_rate = 0.4
    ema_decay = 0.9999

    # Training Config
    epochs = 18
    batch_size = 32
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    print_freq = 100

    # Optimization
    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 0.05

    # Regularization / Augmentation
    mixup_prob = 0.5  # Probability of applying MixUp/CutMix to a batch
    mixup_alpha = 0.8
    cutmix_alpha = 1.0

    # Inference
    tta_steps = 2  # Original + Horizontal Flip


# Ensure output directory exists
os.makedirs(CFG.output_dir, exist_ok=True)
