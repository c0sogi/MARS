import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StreamConfig:
    """
    Configuration for a specific model stream.
    """

    name: str
    arch: str
    weights: str
    input_size: int
    batch_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    interpolation: str = "bicubic"
    local_view_scale: float = 1.28


class Config:
    """
    Global project configuration.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Global Settings ---
    SEED = 42
    NUM_WORKERS = 12  # Matches available vCPUs
    DEVICE = "cuda"  # Assumes GPU availability

    # --- Stream A: ConvNeXt-Large (Baseline) ---
    # Role: Texture-biased, high-performance baseline
    # Weights: torchvision "New Recipe" (IMAGENET1K_V1)
    STREAM_A = StreamConfig(
        name="stream_a",
        arch="convnext_large",
        weights="IMAGENET1K_V1",
        input_size=224,
        batch_size=32,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bicubic",
        local_view_scale=1.28,
    )

    # --- Stream B: RegNetY-128GF (Diversity) ---
    # Role: Massive-scale CNN, SWAG pre-training, high-resolution
    # Weights: torchvision SWAG (IMAGENET1K_SWAG_E2E_V1)
    STREAM_B = StreamConfig(
        name="stream_b",
        arch="regnet_y_128gf",
        weights="IMAGENET1K_SWAG_E2E_V1",
        input_size=384,
        batch_size=8,  # Reduced batch size for large model & resolution
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bicubic",
        local_view_scale=1.28,
    )
