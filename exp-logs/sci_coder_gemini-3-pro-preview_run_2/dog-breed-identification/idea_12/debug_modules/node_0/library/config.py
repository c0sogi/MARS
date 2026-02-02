import os
import torch


class Config:
    """
    Configuration for Dual-Stream Heterogeneous Multi-View Ensemble.
    Stream A: ConvNeXt-Large (CNN)
    Stream B: EVA02-Large (Transformer)
    """

    # -------------------------------------------------------------------------
    # Global Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    # Use CUDA if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute resources
    NUM_WORKERS = 12
    # Batch size for feature extraction (inference).
    # EVA02-Large (448px) is VRAM heavy, but A100 40GB can handle ~32-64.
    # We use a safe default.
    BATCH_SIZE = 32

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Stream A: ConvNeXt-Large (CNN)
    # -------------------------------------------------------------------------
    STREAM_A = {
        "name": "convnext_large",
        "library": "torchvision",
        "weights": "DEFAULT",  # Implies IMAGENET1K_V1
        "input_size": 224,
        # Standard ImageNet normalization
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bicubic",
        # Multi-View Resizing Strategies
        "view_global_size": 224,  # Squish to input size
        "view_standard_resize": 256,  # Resize then CenterCrop(224)
        "view_local_resize": 384,  # Zoom in (Resize larger) then CenterCrop(224)
        "crop_size": 224,
        # Caching
        "cache_prefix": "stream_a_convnext",
    }

    # -------------------------------------------------------------------------
    # Stream B: EVA02-Large (Transformer)
    # -------------------------------------------------------------------------
    STREAM_B = {
        # Specific MIM pre-trained weights
        "name": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "library": "timm",
        "input_size": 448,
        # CLIP/EVA specific normalization
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
        "interpolation": "bicubic",
        # Multi-View Resizing Strategies
        "view_global_size": 448,  # Squish to input size
        "view_standard_resize": 512,  # Resize then CenterCrop(448)
        "view_local_resize": 672,  # Zoom in (~1.5x) then CenterCrop(448)
        "crop_size": 448,
        # Caching
        "cache_prefix": "stream_b_eva02",
    }

    # -------------------------------------------------------------------------
    # Classifier Configuration
    # -------------------------------------------------------------------------
    LOGREG_PARAMS = {
        "Cs": 10,  # Number of C values to try in grid search
        "cv": 5,  # 5-fold cross-validation
        "max_iter": 2000,  # Ensure convergence
        "n_jobs": -1,  # Use all cores
        "random_state": SEED,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "multi_class": "multinomial",
    }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    @staticmethod
    def get_embeddings_path(stream_config, split, view):
        """
        Constructs the path for cached embeddings.
        Args:
            stream_config (dict): STREAM_A or STREAM_B config dict.
            split (str): 'train', 'val', or 'test'.
            view (str): 'global', 'standard', or 'local'.
        """
        filename = f"{stream_config['cache_prefix']}_{split}_{view}_emb.npy"
        return os.path.join(Config.WORKING_DIR, filename)

    @staticmethod
    def get_ids_path(stream_config, split):
        """Constructs the path for cached IDs."""
        filename = f"{stream_config['cache_prefix']}_{split}_ids.npy"
        return os.path.join(Config.WORKING_DIR, filename)

    @staticmethod
    def get_labels_path(stream_config, split):
        """Constructs the path for cached labels."""
        filename = f"{stream_config['cache_prefix']}_{split}_labels.npy"
        return os.path.join(Config.WORKING_DIR, filename)
