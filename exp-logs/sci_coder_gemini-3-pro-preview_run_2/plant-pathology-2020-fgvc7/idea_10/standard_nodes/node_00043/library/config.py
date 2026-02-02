import os
import torch


class CFG:
    """
    Configuration class for Apple Disease Detection.
    Implements the strategy defined in Idea 10:
    Heterogeneous Ensemble (EfficientNetV2-L + ConvNeXt-Base) with GeM Pooling
    and Rank-Calibrated Stacking.
    """

    # ================= General =================
    seed = 42
    debug = False
    n_folds = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading
    num_workers = 4

    # ================= Paths =================
    input_dir = "./input"
    images_dir = os.path.join(input_dir, "images")

    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # Working directory for Idea 10
    working_dir = "./working/idea_10"
    output_dir = working_dir
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Cache paths for deterministic data processing
    train_cache_path = os.path.join(working_dir, "train_cache.parquet")
    val_cache_path = os.path.join(working_dir, "val_cache.parquet")
    test_cache_path = os.path.join(working_dir, "test_cache.parquet")

    # ================= Data =================
    # We decompose the problem into 2 binary tasks: Rust and Scab.
    # Healthy = (1-Rust)*(1-Scab), Multiple = Rust*Scab.
    target_cols = ["rust", "scab"]
    final_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    # ================= Model =================
    # Heterogeneous Ensemble
    backbones = [
        "tf_efficientnetv2_l.in21k_ft_in1k",
        "convnext_base.fb_in22k_ft_in1k_384",
    ]

    # Input resolutions matching backbone design
    img_sizes = {
        "tf_efficientnetv2_l.in21k_ft_in1k": 480,
        "convnext_base.fb_in22k_ft_in1k_384": 384,
    }

    # Generalized Mean Pooling
    gem_p = 3.0
    gem_trainable = True

    # ================= Training =================
    epochs = 15  # Sufficient for convergence with fine-tuning
    batch_size = 4  # Enforced minimum
    learning_rate = 1e-4
    weight_decay = 1e-2
    max_grad_norm = 1000
    use_amp = True
    gradient_accumulation_steps = 4

    # Loss & Optimization
    use_class_weights = True  # Inverse Class Frequency Weights
    label_smoothing = 0.05
    early_stopping_patience = 4

    # ================= Augmentation =================
    # CoarseDropout to force distributed feature learning
    # No CutMix, No ColorJitter
    coarse_dropout_params = {
        "max_holes": 8,
        "max_height": 100,
        "max_width": 100,
        "min_holes": 1,
        "min_height": 16,
        "min_width": 16,
        "fill_value": 0,
        "p": 0.5,
    }

    # ================= Inference =================
    use_tta = True  # Horizontal Flip

    # ================= Stacking =================
    # Rank-Calibrated Stacking
    stacking_method = "rank_calibrated_logistic"
    meta_learner_params = {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    }

    @staticmethod
    def setup():
        """Ensure necessary directories exist."""
        os.makedirs(CFG.working_dir, exist_ok=True)
        os.makedirs(CFG.submission_dir, exist_ok=True)


# Run setup immediately
CFG.setup()
