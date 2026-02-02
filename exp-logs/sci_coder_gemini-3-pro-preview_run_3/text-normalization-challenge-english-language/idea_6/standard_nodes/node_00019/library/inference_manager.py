import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed
from library.symbolic_layer import SymbolicMemory
from library.data_factory import DataFactory, NormalizationDataset
from library.neural_net import MultiTaskSeq2Seq
from torch.utils.data import DataLoader


class CascadePredictor:
    """
    Manages the inference cascade: Symbolic Memory -> Heuristic -> Neural Network.
    """

    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)
        Config.setup_environment()

        # Components
        self.symbolic_memory = SymbolicMemory()
        self.data_factory = DataFactory()
        self.model = None

        # State
        self.tokenizer_loaded = False
        self.model_loaded = False

    def _prepare_resources(self):
        """
        Loads all necessary resources: Stats, Tokenizer, Model.
        """
        print("Preparing resources for inference...")

        # 1. Load Symbolic Memory
        # Try loading from cache first
        try:
            self.symbolic_memory.build_stats(load_cached_data=True)
        except Exception as e:
            print(
                f"Warning: Could not load symbolic stats from cache ({e}). Attempting to build from training data..."
            )
            if os.path.exists(Config.TRAIN_DATA_PATH):
                df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
                self.symbolic_memory.build_stats(df=df_train, load_cached_data=False)
            else:
                raise FileNotFoundError(
                    "Training data not found. Cannot build Symbolic Memory."
                )

        # 2. Load Tokenizer & Encoder
        self.data_factory.load_artifacts()
        if not self.data_factory.tokenizer_fitted:
            raise RuntimeError(
                "Tokenizer not found. Please ensure the training agent has run at least once to generate artifacts."
            )
        self.tokenizer_loaded = True

        # 3. Load Neural Model
        vocab_size = len(self.data_factory.tokenizer)
        self.model = MultiTaskSeq2Seq(vocab_size).to(self.device)

        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
        else:
            print(
                f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Neural inference will fail if attempted."
            )

    def _run_neural_inference(self, df_neural):
        """
        Runs the neural model on a subset of the data.
        """
        if df_neural.empty:
            return {}

        if not self.model_loaded:
            raise RuntimeError(
                "Attempted neural inference but model weights are not loaded."
            )

        print(f"Running neural inference on {len(df_neural)} samples...")

        # Create Dataset and Loader for the subset
        # We use the existing tokenizer
        dataset = NormalizationDataset(
            df_neural,
            self.data_factory.tokenizer,
            self.data_factory.label_encoder,
            mode="test",
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.data_factory.collate_fn,
            pin_memory=True,
        )

        results = {}

        with torch.no_grad():
            for src, raw_txts, ids in loader:
                src = src.to(self.device)

                # Forward pass (Greedy Decoding via teacher_forcing_ratio=0.0)
                # tgt is None, so the model uses its own predictions
                decoder_outputs, _ = self.model(
                    src, tgt=None, teacher_forcing_ratio=0.0
                )

                # decoder_outputs: (batch, max_len, vocab)
                top1 = decoder_outputs.argmax(2)

                # Decode
                for i in range(len(ids)):
                    pred_indices = top1[i]
                    pred_str = self.data_factory.tokenizer.decode(
                        pred_indices, remove_special_tokens=True
                    )
                    results[ids[i]] = pred_str

        return results

    def generate_submission(self, load_cached_data=True):
        """
        Main pipeline to generate predictions for the test set.
        """
        self._prepare_resources()

        # 1. Load Test Data
        # We use process_data to ensure 'prev' and 'next' context columns are added
        print("Loading and processing test data...")
        df_test = self.data_factory.process_data(
            Config.TEST_DATA_PATH,
            "test_processed",
            load_cached_data=load_cached_data,
            is_train_split=False,
        )

        # Ensure string types
        df_test["before"] = df_test["before"].astype(str)
        df_test["prev"] = df_test["prev"].astype(str)
        df_test["next"] = df_test["next"].astype(str)
        df_test["id"] = df_test["id"].astype(str)

        final_predictions = {}
        neural_indices = []

        print("Executing Cascade: Symbolic -> Heuristic -> Neural...")

        # We iterate through the dataframe to apply Symbolic and Heuristic logic
        # Using itertuples for speed
        for row in tqdm(
            df_test.itertuples(index=True), total=len(df_test), desc="Cascade"
        ):
            # row has: Index, sentence_id, token_id, before, id, prev, next

            # 1. Symbolic Memory Lookup
            symbolic_pred = self.symbolic_memory.query(row.prev, row.before, row.next)

            if symbolic_pred is not None:
                final_predictions[row.id] = symbolic_pred
                continue

            # 2. Heuristic Router
            # If OOV (implied by failing step 1) and purely alphabetic, predict Identity
            if row.before.isalpha():
                final_predictions[row.id] = row.before
                continue

            # 3. Neural Candidate
            neural_indices.append(row.Index)

        # 4. Neural Inference
        if neural_indices:
            # Filter dataframe for neural candidates
            df_neural = df_test.loc[neural_indices].copy()
            neural_preds = self._run_neural_inference(df_neural)
            final_predictions.update(neural_preds)

        print(f"Prediction Complete. Total predictions: {len(final_predictions)}")

        # 5. Create Submission File
        # Ensure we cover all IDs in the test set (though logic above should cover all)
        submission_ids = df_test["id"].tolist()

        # Create list in order
        output_data = []
        for uid in submission_ids:
            pred = final_predictions.get(
                uid, ""
            )  # Default to empty if missing (should not happen)
            output_data.append({"id": uid, "after": pred})

        df_submission = pd.DataFrame(output_data)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_submission.to_csv(
            Config.SUBMISSION_PATH, index=False, quoting=1
        )  # quoting=1 is csv.QUOTE_ALL usually, or minimal. Pandas default is fine.

        return df_submission

    def validate(self, load_cached_data=True):
        """
        Runs the cascade on the validation set and calculates accuracy.
        """
        self._prepare_resources()

        print("Loading validation data...")
        df_val = self.data_factory.process_data(
            Config.VAL_DATA_PATH,
            "val_processed",
            load_cached_data=load_cached_data,
            is_train_split=True,  # This filters hard samples for training, but for validation we want full set?
            # Actually, process_data with is_train_split=True filters hard samples.
            # To calculate full accuracy, we need the FULL validation set.
            # But process_data filters it.
            # We should load the raw val parquet to get all samples.
        )

        # Note: DataFactory.process_data filters data if is_train_split=True.
        # We need the full validation set to measure true accuracy.
        # So we load the parquet directly and add context manually using DataFactory's helper.
        df_val_full = pd.read_parquet(Config.VAL_DATA_PATH)
        df_val_full = self.data_factory._add_context(df_val_full)

        # Ensure types
        df_val_full["before"] = df_val_full["before"].astype(str)
        df_val_full["after"] = df_val_full["after"].astype(str)
        df_val_full["prev"] = df_val_full["prev"].astype(str)
        df_val_full["next"] = df_val_full["next"].astype(str)

        correct = 0
        total = len(df_val_full)
        neural_indices = []

        print("Validating Cascade...")

        # We can't update predictions in place easily, so we store them
        predictions = {}

        # 1. Symbolic & Heuristic
        for row in tqdm(
            df_val_full.itertuples(index=True), total=total, desc="Val Cascade"
        ):
            # Symbolic
            res = self.symbolic_memory.query(row.prev, row.before, row.next)
            if res is not None:
                predictions[row.Index] = res
                continue

            # Heuristic
            if row.before.isalpha():
                predictions[row.Index] = row.before
                continue

            # Neural
            neural_indices.append(row.Index)

        # 2. Neural
        if neural_indices:
            df_neural = df_val_full.loc[neural_indices].copy()
            # Need 'id' column for _run_neural_inference mapping
            # df_val_full has 'id' from metadata
            neural_preds_map = self._run_neural_inference(df_neural)

            # Map back to index using id
            # df_neural has 'id'. neural_preds_map keys are 'id'.
            # We need to link 'id' back to 'Index' (row index in df_val_full)
            # Let's build a map id -> Index for neural rows
            id_to_index = {row.id: row.Index for row in df_neural.itertuples()}

            for uid, pred in neural_preds_map.items():
                if uid in id_to_index:
                    predictions[id_to_index[uid]] = pred

        # 3. Calculate Accuracy
        for row in df_val_full.itertuples(index=True):
            pred = predictions.get(row.Index, "")
            if pred == row.after:
                correct += 1

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy}")
        return accuracy
