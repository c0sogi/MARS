import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import setup_logger, GroupStandardizer
from library.data import process_and_cache_graphs, MoleculeDataset, collate_graphs
from library.model import DirectionalMPNN
from library.train import predict_test, set_seed


def generate_submission(
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    debug_sample_size=None,
):
    """
    Generates the submission file using the trained model.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cpu' or 'cuda').
        debug_sample_size (int, optional): Number of molecules to use for debugging.
    """
    # 1. Setup
    logger = setup_logger("Predictor", os.path.join(Config.WORKING_DIR, "predict.log"))
    logger.info("Starting submission generation...")

    set_seed(Config.SEED)

    # 2. Prepare Standardizer
    # We need the standardizer to inverse transform predictions.
    # It attempts to load stats from cache. If missing, we must fit on train data.
    logger.info("Initializing Standardizer...")
    standardizer = GroupStandardizer()

    # We pass the train metadata in case cache is missing and needs to be recomputed.
    # Ideally, training has already run and cached the stats.
    if os.path.exists(Config.TRAIN_META_PATH):
        df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    else:
        # Fallback if train meta is missing (unlikely in this setup)
        df_train_meta = None

    standardizer.fit(df_train_meta, load_cached_data=load_cached_data)

    # 3. Prepare Data
    # Ensure graph data is processed and cached
    cache_paths = process_and_cache_graphs(
        Config.STRUCTURES_CSV, Config.WORKING_DIR, load_cached_data=load_cached_data
    )

    # Create Test Dataset and Loader
    logger.info("Creating Test Loader...")
    test_dataset = MoleculeDataset(
        Config.TEST_META_PATH,
        cache_paths,
        mode="test",
        debug_sample_size=debug_sample_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    # 4. Load Model
    logger.info(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Please train the model first."
        )

    model = DirectionalMPNN(
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_layers=Config.NUM_LAYERS,
        num_radial=Config.NUM_RBF,
        num_spherical=Config.NUM_SBF,
        cutoff=Config.CUTOFF,
        envelope_exponent=Config.ENVELOPE_EXPONENT,
        num_output_layers=Config.NUM_OUTPUT_LAYERS,
        out_emb_dim=Config.TYPE_EMBEDDING_DIM,
    ).to(device)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # 5. Generate Predictions
    # predict_test handles the forward pass and inverse transformation of predictions
    ids, preds = predict_test(model, test_loader, standardizer, device)

    # 6. Save Submission
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub = pd.DataFrame({"id": ids, "scalar_coupling_constant": preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info("Submission generation complete.")
