import os
import torch


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task.
    Centralizes hyperparameters, file paths, and environment settings.
    """

    def __init__(
        self,
        epochs: int = 5,
        train_batch_size: int = 32,
        valid_batch_size: int = 16,
        max_len: int = 96,
        learning_rate: float = 3e-5,
        model_name: str = "roberta-base",
        debug: bool = False,
        seed: int = 42,
    ):
        """
        Initialize configuration with flexible hyperparameters.

        Args:
            epochs (int): Number of training epochs.
            train_batch_size (int): Batch size for training.
            valid_batch_size (int): Batch size for validation.
            max_len (int): Maximum sequence length for tokenization.
            learning_rate (float): Learning rate for the optimizer.
            model_name (str): HuggingFace model backbone name.
            debug (bool): If True, uses a smaller subset of data for debugging.
            seed (int): Random seed for reproducibility.
        """

        # General Settings
        self.SEED = seed
        self.DEBUG = debug

        # Paths - Input (Metadata)
        self.METADATA_DIR = "./metadata"
        self.TRAIN_PATH = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_PATH = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_PATH = os.path.join(self.METADATA_DIR, "test.csv")

        # Paths - Working & Output
        # Ensure ./working/idea_2/ exists as per instructions
        self.WORKING_DIR = "./working/idea_2"
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "cache")
        self.OUTPUT_DIR = "./submission"

        # Create necessary directories
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # File Save Paths
        self.MODEL_SAVE_PATH = os.path.join(self.WORKING_DIR, "best_model.bin")
        self.SUBMISSION_PATH = os.path.join(self.OUTPUT_DIR, "submission.csv")

        # Model Hyperparameters
        self.MODEL_NAME = model_name
        self.MAX_LEN = max_len
        self.EPOCHS = epochs
        self.TRAIN_BATCH_SIZE = train_batch_size
        self.VALID_BATCH_SIZE = valid_batch_size
        self.LEARNING_RATE = learning_rate

        # Hardware Configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 12 vCPUs available; setting workers to a reasonable number
        self.NUM_WORKERS = 0

        # Tokenizer settings
        # Used to load the specific tokenizer matching the model backbone
        self.TOKENIZER_PATH = model_name

    def display(self):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("=" * 30)
