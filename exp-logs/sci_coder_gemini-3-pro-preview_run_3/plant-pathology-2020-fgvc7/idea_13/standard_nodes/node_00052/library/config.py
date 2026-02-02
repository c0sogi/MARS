import os
import torch
import timm


def resolve_model_name(pattern):
    """
    Resolves a model name pattern to a valid timm model identifier.
    Cite debug_lesson_3: Query Registry Keys Programmatically Instead of Guessing.
    Cite debug_lesson_4: Fail Fast on Empty Registry Lookups.
    """
    # Search for models matching the pattern (handling potential suffixes like .in1k)
    matches = timm.list_models(pattern + "*")

    if not matches:
        # Try exact match
        if pattern in timm.list_models():
            matches = [pattern]

    if not matches:
        # Fallback Debugging: List all models in the same family to help diagnosis
        # extracting the family name (e.g., "maxvit" from "maxvit_base_tf_384")
        family = pattern.split("_")[0]
        available_models = timm.list_models(f"*{family}*")

        raise RuntimeError(
            f"Error: No models found matching pattern '{pattern}' in timm registry.\n"
            f"Available models for family '{family}': {available_models[:20]}..."  # Print first 20 to avoid log spam
        )

    # Select the best match
    # Priority 1: Exact match to pattern (if it exists)
    if pattern in matches:
        return pattern

    # Priority 2: Standard ImageNet-1k weights (.in1k suffix)
    in1k_matches = [m for m in matches if m.endswith(".in1k")]
    if in1k_matches:
        # Pick the shortest one (simplest)
        return min(in1k_matches, key=len)

    # Priority 3: First available match
    return matches[0]


class Config:
    """
    Configuration class for the Apple Disease Detection task.
    Implements settings for 'Heterogeneous Ensemble with Decoupled Multi-Task Learning'.
    """

    # =======================
    # General Settings
    # =======================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXPERIMENT_NAME = "idea_13"

    # =======================
    # Directories
    # =======================
    # Input (Read-Only)
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output (Working)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =======================
    # Data Configuration
    # =======================
    NUM_FOLDS = 5
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # =======================
    # Training Hyperparameters
    # =======================
    EPOCHS = 25
    PATIENCE = 10  # Relaxed patience for EMA convergence

    # Optimization
    LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2

    # Hardware
    # A100 40GB allows for decent batch sizes even with large models
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =======================
    # Architecture & Strategy
    # =======================
    # Multi-Task Loss Weight (Lambda)
    # L_total = L_main + LAMBDA * (L_aux_rust + L_aux_scab + L_aux_healthy)
    LAMBDA = 0.5

    # EMA Settings
    USE_EMA = True
    EMA_DECAY = 0.999

    # Model Definitions for Heterogeneous Ensemble
    # Includes Texture Expert (EffNetV2) and Global Expert (MaxViT)
    MODEL_CONFIGS = [
        {
            "name": "effnetv2_m",
            "backbone": "tf_efficientnetv2_m",
            "img_size": 512,  # High res for texture details
            "use_gem": True,
            "gem_p": 3.0,
            "dropout": 0.2,
        },
        {
            # Cite debug_lesson_3: Use a valid, native timm model identifier.
            # Replaced maxvit (unavailable in env) with convnext_tiny (widely supported)
            "name": "convnext_tiny",
            "backbone": "convnext_tiny",
            "img_size": 384,  # Balanced res
            "use_gem": True,
            "gem_p": 3.0,
            "dropout": 0.2,
        },
    ]

    # =======================
    # Inference / TTA
    # =======================
    # Domain-Aware TTA: Horizontal Flip only (leaves have gravity priors)
    TTA_FLIP_HORIZONTAL = True
    TTA_FLIP_VERTICAL = False
    USE_EMA_INFERENCE = True

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"==== Configuration: {cls.EXPERIMENT_NAME} ====")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Models: {[m['name'] for m in cls.MODEL_CONFIGS]}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("==========================================")
