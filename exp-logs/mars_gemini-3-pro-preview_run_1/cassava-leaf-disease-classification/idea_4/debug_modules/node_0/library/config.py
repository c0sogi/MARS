import os
import torch


class Config:
    def __init__(self, debug: bool = False):
        self.debug = debug

        # --- System ---
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 8  # Optimized for 12 vCPUs
        self.print_freq = 100

        # --- Paths ---
        self.input_root = "./input"
        self.metadata_dir = "./metadata"
        self.train_metadata = os.path.join(self.metadata_dir, "train.csv")
        self.val_metadata = os.path.join(self.metadata_dir, "val.csv")
        self.test_metadata = os.path.join(self.metadata_dir, "test.csv")
        self.label_map_path = os.path.join(
            self.input_root, "label_num_to_disease_map.json"
        )

        # Output Directories
        self.working_dir = "./working/idea_4"
        self.output_dir = os.path.join(self.working_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Data ---
        self.num_classes = 5

        # --- Model ---
        # Using ConvNeXt-Base pre-trained on ImageNet-21k
        self.model_name = "convnext_base.fb_in22k"
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0  # Explicitly 0.0 as per strategy
        self.use_ema = True
        self.ema_decay = 0.9999

        # --- Optimization ---
        self.weight_decay = 0.05
        self.layer_decay = 0.8  # Layer-wise Learning Rate Decay (LLRD)
        self.opt_eps = 1e-8
        self.opt_betas = (0.9, 0.999)
        self.clip_grad = 5.0  # Gradient clipping

        # --- Augmentation ---
        self.mixup_alpha = 0.8
        self.cutmix_alpha = 1.0
        self.mixup_prob = 1.0  # Probability to apply Mixup or Cutmix
        self.mixup_switch_prob = 0.5  # Probability to switch between Mixup and Cutmix
        self.aug_prob = 0.5  # Probability for geometric augmentations

        # --- Training Phase 1: Coarse (384x384) ---
        self.img_size_coarse = 384
        self.batch_size_coarse = 32
        self.grad_accum_steps_coarse = 1  # Effective Batch Size = 32

        self.epochs_warmup = 1
        self.epochs_coarse = 12
        self.lr_coarse = 2e-4
        self.min_lr_coarse = 1e-6

        # --- Training Phase 2: Fine-tuning (512x512) ---
        self.img_size_fine = 512
        self.batch_size_fine = 16
        self.grad_accum_steps_fine = 2  # Effective Batch Size = 32

        self.epochs_fine = 5
        self.lr_fine = 2e-5  # Lower LR for fine-tuning
        self.min_lr_fine = 1e-7

        # --- Inference ---
        self.tta_steps = 3  # Original + Horizontal Flip + Vertical Flip

        # --- Debug Overrides ---
        if self.debug:
            self.epochs_warmup = 0
            self.epochs_coarse = 1
            self.epochs_fine = 1
            self.debug_subset_size = 100  # Only use 100 samples for debugging
