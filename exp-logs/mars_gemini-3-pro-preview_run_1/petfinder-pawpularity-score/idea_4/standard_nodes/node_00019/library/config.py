import os


class Config:
    """
    Configuration class for the Multi-Paradigm Stacking Ensemble project.
    Stores file paths, hyperparameters, and model specifications.
    """

    # =========================================================================
    # Global Hyperparameters
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32  # Adjusted for A100 GPU memory with Large models
    NUM_WORKERS = 4

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate features (idea_4 specific)
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output path for final submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Specifications
    # =========================================================================
    # We define specific configurations for each expert in the ensemble.
    # Note: CLIP uses different normalization stats than standard ImageNet models.

    MODEL_CONFIGS = {
        "swin_large": {
            "model_name": "swin_large_patch4_window7_224",  # Swin Transformer Large
            "img_size": 224,
            "mean": (0.485, 0.456, 0.406),  # ImageNet default
            "std": (0.229, 0.224, 0.225),  # ImageNet default
            "use_clip_norm": False,
        },
        "convnext_large": {
            "model_name": "convnext_large.fb_in1k",  # ConvNeXt Large (ImageNet-1k)
            "img_size": 224,
            "mean": (0.485, 0.456, 0.406),  # ImageNet default
            "std": (0.229, 0.224, 0.225),  # ImageNet default
            "use_clip_norm": False,
        },
        "clip_large": {
            "model_name": "vit_large_patch14_clip_224.openai",  # CLIP ViT-L/14
            "img_size": 224,
            "mean": (0.48145466, 0.4578275, 0.40821073),  # OpenAI CLIP mean
            "std": (0.26862954, 0.26130258, 0.27577711),  # OpenAI CLIP std
            "use_clip_norm": True,
        },
    }

    # =========================================================================
    # Training / Stacking Parameters
    # =========================================================================
    # Ridge Regression Alphas for Level-0 Experts
    # Using a wide range to allow RidgeCV to find the optimal regularization
    RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    # Metadata columns to include in the feature vector
    METADATA_COLS = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
