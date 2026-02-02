import os
import torch
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# Global Configuration & Paths
# =============================================================================
SEED = 42
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 11
WORKING_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Hardware settings
# 12 vCPUs available
NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class DataConfig:
    """Configuration for data loading and paths."""

    train_csv: str = os.path.join(METADATA_DIR, "train.csv")
    val_csv: str = os.path.join(METADATA_DIR, "val.csv")
    test_csv: str = os.path.join(METADATA_DIR, "test.csv")
    # Binary classification: Dog (1) vs Cat (0)
    num_classes: int = 1


@dataclass
class TrainConfig:
    """Configuration for the training loop."""

    epochs: int = 10
    # Gradient accumulation steps (1 = no accumulation)
    accumulate_grad_batches: int = 1
    early_stopping_patience: int = 3
    # Optimization
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    min_lr: float = 1e-6
    # Loss function settings
    use_label_smoothing: bool = False  # Explicitly disabled per Idea 11 strategy


@dataclass
class ModelConfig:
    """Configuration for a specific model architecture."""

    model_name: str
    input_size: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 1e-2
    # Multi-Sample Dropout settings
    use_multi_sample_dropout: bool = True
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    # TTA settings
    use_tta: bool = True


# =============================================================================
# Ensemble Definitions (Triple Heterogeneous Ensemble)
# =============================================================================

# 1. ResNet-50: The "Standard CNN" anchor.
# Resolution: 256x256 for fine spatial details.
resnet_config = ModelConfig(
    model_name="resnet50.a1_in1k", input_size=256, batch_size=64, learning_rate=1e-4
)

# 2. ConvNeXt-Small: The "Modern CNN".
# Resolution: 256x256.
convnext_config = ModelConfig(
    model_name="convnext_small.fb_in1k",
    input_size=256,
    batch_size=32,
    learning_rate=1e-4,
)

# 3. MaxViT-Tiny: The "Multi-Axis Transformer".
# Resolution: 224x224 (Native grid size).
maxvit_config = ModelConfig(
    model_name="maxvit_tiny_tf_224.in1k",
    input_size=224,
    batch_size=32,
    learning_rate=1e-4,
)

# List of models to train
MODEL_CONFIGS = [resnet_config, convnext_config, maxvit_config]

# Shared instances
data_config = DataConfig()
train_config = TrainConfig()
