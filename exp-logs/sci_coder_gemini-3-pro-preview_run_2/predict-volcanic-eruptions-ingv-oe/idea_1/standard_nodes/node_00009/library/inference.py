import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from library.config import Config
from library.feature_extractor import extract_features, extract_spectrograms
from library.model import VolcanoHybrid, set_seed
from library.data_loader import VolcanoDataset


def generate_predictions(
    model=None,
    device=None,
    debug_size=None,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    if device is None:
        device = torch.device(Config.DEVICE)

    set_seed(Config.SEED)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print("Starting inference pipeline...")

    # 1. Load Features
    df_test = extract_features(
        Config.TEST_METADATA_PATH,
        Config.TEST_FEATURES_CACHE,
        load_cached_data,
        debug_size,
    )
    specs_test = extract_spectrograms(
        Config.TEST_METADATA_PATH, Config.TEST_SPEC_CACHE, load_cached_data, debug_size
    )

    # 2. Prepare Tabular
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in df_test.columns if c not in exclude_cols]
    X_test = df_test[feature_cols].values.astype(np.float32)
    segment_ids = df_test["segment_id"].values
    input_dim = len(feature_cols)

    # 3. Scale Tabular
    if os.path.exists(Config.SCALER_MEAN_PATH):
        mean = np.load(Config.SCALER_MEAN_PATH)
        scale = np.load(Config.SCALER_SCALE_PATH)
        X_test = ((X_test - mean) / scale).astype(np.float32)

    # 4. Scale Spectrograms
    if os.path.exists(Config.SPEC_MEAN_PATH):
        s_mean = np.load(Config.SPEC_MEAN_PATH)
        s_std = np.load(Config.SPEC_STD_PATH)
        specs_test = ((specs_test - s_mean) / (s_std + 1e-6)).astype(np.float32)

    # 5. Load Model
    if model is None:
        model = VolcanoHybrid(
            tabular_input_dim=input_dim,
            hidden_layers=Config.HIDDEN_LAYERS,
            dropout_rate=Config.DROPOUT_RATE,
        ).to(device)
        if os.path.exists(Config.MODEL_SAVE_PATH):
            model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=device)
            )

    model.eval()
    predictions = []

    # Use VolcanoDataset for consistent loading (augment=False)
    test_dataset = VolcanoDataset(specs_test, X_test, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for inputs in test_loader:
            spec, tab = inputs
            spec = spec.to(device)
            tab = tab.to(device)

            outputs = model((spec, tab)).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            predictions.extend(outputs.cpu().numpy())

    # 7. Inverse Transform Predictions (Cite solution_lesson_node_00001)
    if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
        Config.TARGET_STD_PATH
    ):
        t_mean = np.load(Config.TARGET_MEAN_PATH)
        t_std = np.load(Config.TARGET_STD_PATH)
        predictions = np.array(predictions) * t_std + t_mean
    else:
        print("Warning: Target scaler not found. Predictions are in scaled space.")

    # 8. Generate Submission File
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return df_sub
