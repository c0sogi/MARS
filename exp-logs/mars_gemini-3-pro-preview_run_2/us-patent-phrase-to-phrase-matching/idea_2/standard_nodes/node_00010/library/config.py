import os
import torch


class Config:
    """
    Configuration for the Phrase Similarity Task.
    Implements settings for the Context-Enriched DeBERTa-v3-Large Cross-Encoder.
    """

    # =========================================================================
    # General Configuration
    # =========================================================================
    seed = 42
    n_folds = 5

    # Debugging: Set to True to train on a small subset for verification
    debug = False
    debug_sample_size = 200

    # =========================================================================
    # Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"  # Using generated metadata with stratified splits

    # Directory for caching processed datasets and saving models
    working_dir = "./working/idea_2"

    # Directory for final submission
    submission_dir = "./submission"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"

    # We treat the problem as a 5-class classification task
    # Classes correspond to scores: {0.0, 0.25, 0.5, 0.75, 1.0}
    num_classes = 5

    # Mapping from class index to scalar score for Expected Value calculation
    score_map = {0: 0.00, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    max_length = 128  # Sufficient for Context + Anchor + Target
    use_cpc_text = True  # Flag to enable mapping CPC codes to textual descriptions

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 5

    # Batch sizes tailored for A100 40GB with Large model
    train_batch_size = 8
    valid_batch_size = 16

    # Optimization
    learning_rate = 2e-5
    min_lr = 1e-6
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # Loss Function
    label_smoothing = 0.1  # Helps with noisy expert annotations

    # =========================================================================
    # Hardware & Performance
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    use_fp16 = True  # Mixed Precision Training

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Initialize directories upon import
Config.setup()
