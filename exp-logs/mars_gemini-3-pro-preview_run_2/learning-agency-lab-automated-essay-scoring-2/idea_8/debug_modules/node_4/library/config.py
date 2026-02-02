import os
import torch

# Ensure necessary directories exist
OUTPUT_DIR = "./working/idea_8/"
SUBMISSION_DIR = "./submission/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


class CFG:
    # --- General ---
    debug = False  # Set to True for fast debugging runs
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    # Using the metadata files generated in the previous step
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    output_dir = OUTPUT_DIR
    submission_file = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model: Deep Semantic Branch ---
    model_name = "microsoft/deberta-v3-large"
    max_length = 1024
    gradient_checkpointing = True

    # Pooling Strategy
    pool_type = "weighted_layer_pool"  # Options: 'mean', 'cls', 'weighted_layer_pool'
    num_layers_pool = 4  # Number of last layers to aggregate

    # Training Hyperparameters
    num_folds = 5
    epochs = 4
    train_batch_size = 4  # Adjusted for A100 memory with large sequence length
    valid_batch_size = 8
    learning_rate = 1e-5
    weight_decay = 0.01
    scheduler = "cosine"  # Cosine annealing
    min_lr = 1e-7
    eps = 1e-6
    betas = (0.9, 0.999)
    max_grad_norm = 1000

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 2  # Start AWP after the model has stabilized
    awp_eps = 1e-4
    awp_lr = 1e-4

    # --- Model: Lexical & Morphological Branches (Ridge) ---
    # Lexical (Word N-grams)
    word_ngram_range = (1, 3)
    word_min_df = 3

    # Morphological (Char N-grams)
    char_ngram_range = (3, 5)
    char_min_df = 3

    # Ridge Regression Settings
    ridge_alpha = 1.0

    # --- Model: Meta-Learner (LightGBM) ---
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_estimators": 1000,
        "random_state": 42,
    }

    # --- Post-Processing ---
    # Nelder-Mead optimization for thresholding
    optimize_thresholds = True
