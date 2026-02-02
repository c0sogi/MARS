import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import os

from library.config import Config, seed_everything
from library.utils import load_checkpoint, write_submission
from library.data_processor import load_data
from library.dataset import ChatbotDataset
from library.model import SiameseDeberta
from library.trainer import Trainer


def run_inference(
    load_cached_data=True,
    debug=False,
    batch_size=Config.VALID_BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Executes the inference pipeline for the Siamese DeBERTa model.

    Steps:
    1. Loads processed test data (text + scaled meta-features).
    2. Initializes the tokenizer and creates the test DataLoader.
    3. Loads the model architecture and restores weights from the best checkpoint.
    4. Generates predictions using the Trainer class.
    5. Writes the submission file to disk.

    Args:
        load_cached_data (bool): If True, attempts to load data from Parquet cache.
        debug (bool): If True, runs on a small subset of data.
        batch_size (int): Batch size for the DataLoader.
        device (torch.device): Device to perform inference on.

    Returns:
        None
    """
    # 1. Setup
    seed_everything(Config.SEED)
    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # load_data returns (train, val, test). We only need test.
    # The data_processor handles text cleaning and meta-feature scaling.
    print("Loading test data...")
    _, _, df_test = load_data(load_cached_data=load_cached_data, debug=debug)

    # 3. Prepare Tokenizer and Dataset
    print(f"Initializing tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print("Creating test dataset and dataloader...")
    test_dataset = ChatbotDataset(
        df=df_test, tokenizer=tokenizer, max_length=Config.MAX_LENGTH, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Load Model
    print("Initializing model architecture...")
    # meta_dim=3 corresponds to [len_prompt, len_a, len_b]
    model = SiameseDeberta(model_name=Config.MODEL_NAME, num_classes=3, meta_dim=3)

    print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        epoch, val_loss = load_checkpoint(
            model, path=Config.MODEL_SAVE_PATH, device=device
        )
        print(f"Successfully loaded model (Epoch: {epoch}, Val Loss: {val_loss})")
    else:
        print(
            f"Warning: Checkpoint not found at {Config.MODEL_SAVE_PATH}. Using initialized weights."
        )

    # 5. Generate Predictions
    print("Generating predictions...")
    trainer = Trainer(model, device=device)
    predictions = trainer.predict(test_loader)

    # 6. Write Submission
    print(f"Writing submission file to {Config.SUBMISSION_PATH}...")
    write_submission(
        ids=df_test["id"].values,
        predictions=predictions,
        output_path=Config.SUBMISSION_PATH,
    )

    print("Inference pipeline completed successfully.")
