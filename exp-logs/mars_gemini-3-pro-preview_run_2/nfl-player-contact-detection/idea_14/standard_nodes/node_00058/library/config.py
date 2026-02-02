import os
import torch


class Config:
    def __init__(self, debug=False, sample_size=None, epochs=None):
        """
        Configuration for the NFL Contact Detection Task (EC-GRN-v2).

        Args:
            debug (bool): If True, runs in debug mode with a smaller dataset and fewer epochs.
            sample_size (int, optional): Number of samples to use in debug mode.
            epochs (int, optional): Override the number of training epochs.
        """
        # --- File Paths ---
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_14"
        self.SUBMISSION_DIR = "./submission"

        # Ensure output directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # --- Data Preprocessing ---
        # Temporal Window: +/- 5 steps (t-5 ... t ... t+5) -> 11 steps total
        self.WINDOW_SIZE = 5
        self.TOTAL_STEPS = 2 * self.WINDOW_SIZE + 1

        # Feature Selection
        self.TRACKING_COLS = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "sa",
        ]

        # Categorical Features for Entity Embeddings
        self.CAT_COLS = ["position", "team"]

        # Feature Engineering Flags (specific to EC-GRN-v2 logic)
        self.IMPUTE_GROUND_VELOCITY_ZERO = True  # Force Ground velocity/accel to 0
        self.USE_LOG_DISTANCE = True  # Apply log1p to distance
        self.CLAMP_CLOSING_SPEED = True  # Stabilize closing speed calc

        # --- Model Architecture ---
        # Wide-Format Gated Residual Network
        self.HIDDEN_SIZE = 512
        self.NUM_BLOCKS = 3
        self.DROPOUT = 0.1

        # Embedding Dimensions: (Num Categories, Embedding Size)
        # Position: ~28 positions -> dim 8
        # Team: 2 teams (home/away) + unknown -> dim 2
        self.EMBEDDING_DIMS = {"position": (32, 8), "team": (4, 2)}

        # --- Training Hyperparameters ---
        self.SEED = 42
        self.BATCH_SIZE = 4096
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-2

        # Focal Loss Parameters
        self.FOCAL_ALPHA = 0.25
        self.FOCAL_GAMMA = 2.0

        # Optimization
        self.PATIENCE = 3  # Early stopping patience
        self.EPOCHS = epochs if epochs is not None else 15

        # --- Hardware ---
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 4

        # --- Debugging / Runtime Control ---
        self.DEBUG = debug
        if self.DEBUG:
            self.SAMPLE_SIZE = sample_size if sample_size is not None else 10000
            self.EPOCHS = 2  # Shorten training for debug

            # Ensure batch size does not exceed sample size to prevent empty DataLoaders with drop_last=True
            if self.BATCH_SIZE > self.SAMPLE_SIZE:
                self.BATCH_SIZE = self.SAMPLE_SIZE

            print(
                f"DEBUG MODE ENABLED: Sample Size={self.SAMPLE_SIZE}, Epochs={self.EPOCHS}"
            )
        else:
            self.SAMPLE_SIZE = None  # Use full dataset

    def display(self):
        """Helper to print the current configuration."""
        print("\n" + "=" * 20 + " CONFIGURATION " + "=" * 20)
        for k, v in self.__dict__.items():
            if not k.startswith("__"):
                print(f"{k}: {v}")
        print("=" * 55 + "\n")
