import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import warnings
import logging
from transformers import logging as hf_logging

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_factory import create_dataloaders, get_tokenizer
from library.modeling import InsultModel
from library.pretrain_mlm import run_domain_adaptation
from library.train_supervised import train_seed
from library.inference import generate_submission


def main():
    # 1. Configuration and Setup
    print("Setting up demonstration environment...")

    # Suppress verbose output for cleaner execution
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Define working directory for demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config for fast execution and resource optimization
    print("Overriding Config for fast execution...")
    Config.working_dir = demo_dir
    Config.cache_dir = os.path.join(demo_dir, "cache")
    Config.dapt_model_output_dir = os.path.join(demo_dir, "dapt_model")
    Config.submission_dir = os.path.join(demo_dir, "submission")
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Use a smaller model for speed (Base vs Large)
    Config.model_name = "roberta-base"
    Config.tokenizer_name = "roberta-base"

    # Reduce hyperparameters for demo speed
    Config.max_length = 32
    Config.dapt_epochs = 1
    Config.sft_epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.dapt_batch_size = 4
    Config.seeds = [42]  # Run only one seed
    Config.freeze_encoder_layers = 2  # Freeze fewer layers for base model

    # Set device (CPU is sufficient for this tiny batch, but use CUDA if available)
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {Config.device}")

    # 2. Prepare Subset Data
    # Create tiny subsets of the data to ensure the pipeline runs instantly
    print("Creating data subsets...")
    try:
        # Load original metadata
        df_train = pd.read_csv("./metadata/train.csv")
        df_val = pd.read_csv("./metadata/val.csv")
        df_test = pd.read_csv("./metadata/test.csv")

        # Take top 10 rows
        sub_train = df_train.head(10)
        sub_val = df_val.head(10)
        sub_test = df_test.head(10)

        # Save to demo directory
        train_path = os.path.join(demo_dir, "train.csv")
        val_path = os.path.join(demo_dir, "val.csv")
        test_path = os.path.join(demo_dir, "test.csv")

        sub_train.to_csv(train_path, index=False)
        sub_val.to_csv(val_path, index=False)
        sub_test.to_csv(test_path, index=False)

        # Update Config paths to point to these new files
        Config.train_path = train_path
        Config.val_path = val_path
        Config.test_path = test_path

    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

    # 3. Demonstrate Data Loading
    print("\n[Step 1] Verifying Data Loading...")
    tokenizer = get_tokenizer()

    # Create supervised dataloader
    # load_cached_data=False ensures we process our new subset files
    dl_supervised = create_dataloaders("supervised", tokenizer, load_cached_data=False)
    batch = next(iter(dl_supervised))

    # Verify batch structure and shapes
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "attention_mask" in batch, "Batch missing attention_mask"
    assert "labels" in batch, "Batch missing labels"
    assert batch["input_ids"].shape == (
        Config.train_batch_size,
        Config.max_length,
    ), f"Unexpected shape: {batch['input_ids'].shape}"
    print("Data loading successful.")

    # 4. Demonstrate Model Initialization & Forward Pass
    print("\n[Step 2] Verifying Model Architecture...")
    model = InsultModel(pretrained=True)
    model.to(Config.device)
    model.eval()

    with torch.no_grad():
        inputs = batch["input_ids"].to(Config.device)
        mask = batch["attention_mask"].to(Config.device)
        logits = model(inputs, mask)

    assert logits.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), f"Logits shape mismatch: {logits.shape}"
    print("Model forward pass successful.")

    # Clean up model to free memory
    del model, batch, inputs, mask, logits
    torch.cuda.empty_cache()

    # 5. Demonstrate Domain-Adaptive Pre-training (DAPT)
    print("\n[Step 3] Running Domain-Adaptive Pre-training (DAPT)...")
    # This function trains on the MLM objective and saves the model
    run_domain_adaptation()

    # Verify DAPT output
    assert os.path.exists(
        Config.dapt_model_output_dir
    ), "DAPT output directory not created"
    # Check for config.json or pytorch_model.bin/model.safetensors
    assert (
        len(os.listdir(Config.dapt_model_output_dir)) > 0
    ), "DAPT model files not found"
    print("DAPT completed successfully.")

    # 6. Demonstrate Supervised Fine-Tuning
    print("\n[Step 4] Running Supervised Fine-Tuning...")
    # This trains the model on the labeled data and saves it to Config.working_dir
    train_seed(42)

    model_path = os.path.join(Config.working_dir, "model_seed_42.bin")
    assert os.path.exists(model_path), f"Trained model not found at {model_path}"
    print("Supervised training completed successfully.")

    # 7. Demonstrate Inference and Submission
    print("\n[Step 5] Generating Submission...")
    # This generates predictions using the trained model and creates the submission CSV
    generate_submission()

    assert os.path.exists(Config.submission_path), "Submission file not found"

    # Load submission to verify contents
    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")

    # Verify predictions
    # Note: Since we only predicted on 10 samples, the inference script (which loads the full sample submission)
    # will only update the first 10 rows. We verify those.
    preds = sub_df["Insult"].values[:10]

    # Check range [0, 1]
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions out of range [0, 1]"

    # Check that we have non-zero predictions (confirming model inference actually happened)
    # Sigmoid output is rarely exactly 0 unless logits are extremely negative.
    if np.all(preds == 0):
        print(
            "Warning: All predictions are exactly 0. This is technically valid but statistically unlikely for initialized weights."
        )
    else:
        print("Predictions contain non-zero values, confirming inference ran.")

    print("\n[Success] All pipeline components verified.")


if __name__ == "__main__":
    main()
