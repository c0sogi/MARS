import os
import torch


class Config:
    """
    Configuration class for Deep-Stem ResNet-34 (Non-Anti-Aliased) Ensemble Distillation.
    """

    def __init__(self, debug=False, output_dir="./working/idea_26"):
        # =================================================================
        # 1. Environment & Paths
        # =================================================================
        self.DEBUG = debug
        self.SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = 4  # Number of dataloader workers

        # Input Paths
        self.INPUT_ROOT = "./input"
        self.METADATA_DIR = "./metadata"
        self.TRAIN_METADATA = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_METADATA = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_METADATA = os.path.join(self.METADATA_DIR, "test.csv")

        self.SPECTROGRAM_DIR = os.path.join(
            self.INPUT_ROOT, "supplemental_data", "spectrograms"
        )

        # Output Paths
        self.OUTPUT_DIR = output_dir
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.SUBMISSION_DIR = "./submission"
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # =================================================================
        # 2. Model Architecture
        # =================================================================
        # Using resnet34d (Deep Stem) and explicitly disabling anti-aliasing (BlurPool)
        self.MODEL_NAME = "resnet34d"
        self.MODEL_PARAMS = {
            "pretrained": True,
            "num_classes": 19,
            "in_chans": 3,
            "global_pool": "avg",
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        }

        # =================================================================
        # 3. Data Preprocessing
        # =================================================================
        # High-Fidelity Resolution: 256 (Height/Freq) x 640 (Width/Time)
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 640
        self.IMG_SIZE = (self.IMG_HEIGHT, self.IMG_WIDTH)

        # ImageNet Normalization
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        # =================================================================
        # 4. Training Hyperparameters
        # =================================================================
        self.BATCH_SIZE = 32

        # Learning Rate
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-4

        # Epochs
        # If debug is True, run for very few epochs to verify pipeline
        self.EPOCHS = 2 if self.DEBUG else 50

        # SWA Configuration
        # Teacher: SWA last 25% (approx start at epoch 37)
        self.SWA_START_EPOCH_TEACHER = 1 if self.DEBUG else 37
        # Student: SWA last 30% (approx start at epoch 35)
        self.SWA_START_EPOCH_STUDENT = 1 if self.DEBUG else 35

        self.SWA_LR = 1e-3  # High constant LR for SWA phase
        self.SWA_ANNEAL_EPOCHS = 3
        self.SWA_ANNEAL_STRATEGY = "cos"

        # Mixup
        self.MIXUP_ALPHA = 0.2
        self.MIXUP_PROB = 1.0  # Probability of applying mixup

        # Ensemble Config
        self.NUM_TEACHERS = 3

    def get_model_save_path(self, name):
        """Helper to get full path for saving model checkpoints."""
        return os.path.join(self.OUTPUT_DIR, f"{name}.pth")

    def __repr__(self):
        return str(self.__dict__)
