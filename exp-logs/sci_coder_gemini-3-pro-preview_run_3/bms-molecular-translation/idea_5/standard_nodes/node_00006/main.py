import os
import sys
import torch
import pandas as pd
import numpy as np
import random
import nltk

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import levenshtein_distance
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms
from library.model import ResNetTCN
from library.trainer import train
from library.inference import generate_submission, greedy_decode
from torch.utils.data import DataLoader


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_pipeline():
    # --- 1. Configuration & Setup ---
    print("--- Configuring Pipeline ---")
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10000  # Train on 10k samples for speed
    Config.NUM_EPOCHS = 2  # 2 Epochs to prove learning
    Config.BATCH_SIZE = 256  # High batch size for A100
    Config.NUM_WORKERS = 8  # Utilize vCPUs
    Config.PATIENCE = 2  # Strict early stopping

    Config.print_config()

    # --- 2. Training ---
    print("\n--- Starting Training ---")
    # train() handles tokenizer build, loader setup, and training loop
    # It saves the best model to Config.CHECKPOINT_PATH
    trainer = train(load_cached_data=False)

    # --- 3. Validation Assessment & Failure Analysis ---
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Tokenizer (cached from training)
    tokenizer = Tokenizer()
    tokenizer.build_vocab(load_cached_data=True)

    # Setup Validation Loader (using DEBUG subset as configured)
    val_transform = get_transforms("valid")
    val_dataset = InChiDataset(Config.VAL_METADATA, tokenizer, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    device = Config.DEVICE
    model = ResNetTCN(vocab_size=tokenizer.get_vocab_size())
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    predictions = []
    ground_truths = []
    lengths = []

    print(f"Validating on {len(val_dataset)} samples...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Run Inference
            batch_preds = greedy_decode(
                model, images, tokenizer, max_len=Config.MAX_LEN, device=device
            )

            # Decode Ground Truth
            for i in range(len(labels)):
                gt_seq = labels[i]
                gt_text = tokenizer.sequence_to_text(gt_seq)

                predictions.append(batch_preds[i])
                ground_truths.append(gt_text)
                lengths.append(len(gt_text))

    # Compute Metric
    lev_distances = []
    for p, t in zip(predictions, ground_truths):
        lev_distances.append(levenshtein_distance(p, t))

    final_metric = np.mean(lev_distances)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    analysis_df = pd.DataFrame({"length": lengths, "error": lev_distances})

    correlation = analysis_df["length"].corr(analysis_df["error"])
    print(f"Correlation between InChI Length and Error: {correlation:.4f}")

    if correlation > 0.3:
        print("Insight: The model struggles significantly more with longer sequences.")

    # --- 4. Submission Generation ---
    print("\n--- Generating Submission ---")
    # Disable Debug to process the FULL test set
    Config.DEBUG = False

    # Generate submission using the library function
    # This will reload the test dataset (full size) and the model
    generate_submission(load_cached_data=True)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    run_pipeline()
