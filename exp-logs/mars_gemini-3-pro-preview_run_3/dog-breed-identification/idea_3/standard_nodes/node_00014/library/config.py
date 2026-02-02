import os
import torch


class Config:
    # Reproducibility
    seed = 42

    # Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_3"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    cache_dir = working_dir
    best_model_path = os.path.join(working_dir, "best_model.pth")
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure submission directory exists
    os.makedirs(submission_dir, exist_ok=True)

    # Data
    img_size = 224
    num_classes = 120
    # Use a smaller batch size for the larger model/resolution to fit in GPU memory
    batch_size = 64
    num_workers = 4

    # Model
    model_name = "convnext_small"
    pretrained = True

    # Training Hyperparameters
    epochs = 30
    warmup_epochs = 1

    # Learning Rates
    lr_warmup = 1e-3  # For the linear head training phase
    lr_fine_tune = 1e-5  # For the full fine-tuning phase
    weight_decay = 1e-4

    # Scheduler
    scheduler_t_max = 30

    # Debugging
    # Set to True to run on a small subset of data for quick verification
    debug = False
    debug_sample_size = 100

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        # Print config for verification
        print(f"Config initialized with device: {self.device}")
        print(f"Model: {self.model_name}, Image Size: {self.img_size}")
