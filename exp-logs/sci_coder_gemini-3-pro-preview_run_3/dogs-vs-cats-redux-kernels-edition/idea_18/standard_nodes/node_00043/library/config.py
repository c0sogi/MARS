import os
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Global Directory & Path Setup
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_18"
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Global Training Constants
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda"
NUM_WORKERS = 4
DEBUG = False  # Set to True to run on a small subset for debugging purposes


# -----------------------------------------------------------------------------
# Model Configuration Dataclass
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Configuration for a specific model architecture in the heterogeneous ensemble.
    """

    model_name: str
    img_size: int
    epochs: int
    batch_size: int
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_folds: int = 5
    seed: int = SEED

    @property
    def name(self) -> str:
        """
        Returns a unique identifier for the model configuration, used for saving checkpoints.
        Replaces dots with underscores to be filesystem-friendly.
        """
        clean_name = self.model_name.replace(".", "_")
        return f"{clean_name}_{self.img_size}"


# -----------------------------------------------------------------------------
# Architecture-Specific Configurations
# -----------------------------------------------------------------------------

# 1. ResNet-50: The "Standard CNN" anchor.
# Using 'resnet50.a1_in1k' for modern training recipe weights.
# Resolution: 256x256
RESNET_CFG = ModelConfig(
    model_name="resnet50.a1_in1k",
    img_size=256,
    epochs=8,
    batch_size=64,  # ResNet is efficient, can handle larger batches
)

# 2. ConvNeXt-Small: The "Modern CNN".
# Resolution: 288x288 to capture fine-grained details.
CONVNEXT_CFG = ModelConfig(
    model_name="convnext_small.fb_in1k",
    img_size=288,
    epochs=8,
    batch_size=32,
)

# 3. MaxViT-Tiny: The "Multi-Axis Transformer".
# Resolution: 224x224 to match native grid structure.
# Epochs: 15 (Extended schedule for convergence).
MAXVIT_CFG = ModelConfig(
    model_name="maxvit_tiny_tf_224.in1k",
    img_size=224,
    epochs=15,
    batch_size=32,
)

# List of all configurations for the ensemble loop
ENSEMBLE_CONFIGS = [RESNET_CFG, CONVNEXT_CFG, MAXVIT_CFG]
