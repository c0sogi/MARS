import os
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config


class FeatureExtractor:
    """
    Handles the extraction and caching of embeddings for the DC-AN model.
    Uses 'sentence-transformers/all-mpnet-base-v2' to encode code and markdown cells.
    """

    def __init__(self):
        """
        Initialize the feature extractor with the backbone model.
        """
        Config.set_seed(Config.SEED)
        self.device = Config.DEVICE

        print(f"Initializing FeatureExtractor with backbone: {Config.BACKBONE_NAME}")
        self.model = SentenceTransformer(Config.BACKBONE_NAME, device=self.device)
        self.model.max_seq_length = Config.MAX_LENGTH

    def _get_notebook_data(self, filepath):
        """
        Reads a JSON notebook file safely.
        """
        full_path = os.path.join(Config.INPUT_DIR, filepath)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            return None

    def _generate_labels(self, cell_order, cell_types):
        """
        Generates labels for markdown cells based on the ground truth order.
        Label = Number of code cells that appear before the markdown cell.
        This corresponds to the insertion index in the sequence of code anchors.
        """
        if not isinstance(cell_order, str):
            return {}

        order_list = cell_order.split()
        labels = {}
        code_count = 0

        for cell_id in order_list:
            ctype = cell_types.get(cell_id, "unknown")
            if ctype == "code":
                code_count += 1
            elif ctype == "markdown":
                labels[cell_id] = code_count

        return labels

    def process_dataset(
        self, metadata_path, output_path, load_cached_data=True, is_test=False
    ):
        """
        Main function to process a dataset (train, val, or test).
        Extracts embeddings and labels, then saves to Parquet.
        """
        # 1. Caching Check
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}...")
            return pd.read_parquet(output_path, engine="pyarrow")

        print(f"Processing data from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        # Debugging sample size
        if Config.DEBUG_SAMPLE_SIZE is not None:
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)
            print(f"Debug mode: Processing {len(df_meta)} samples.")

        # Buffers for batch processing
        text_buffer = []
        meta_buffer = []  # List of tuples: (batch_index, cell_type, cell_id)
        current_batch_records = {}  # batch_index -> record_dict

        # Output containers
        all_ids = []
        all_code_embs = []
        all_md_embs = []
        all_md_labels = []
        all_md_ids = []

        batch_counter = 0
        BATCH_SIZE_NBS = (
            200  # Accumulate 200 notebooks before encoding to save memory/time overhead
        )

        def flush_buffer():
            """Encodes buffered texts and distributes embeddings back to records."""
            nonlocal text_buffer, meta_buffer, current_batch_records, batch_counter

            if not text_buffer:
                return

            # Encode all texts in the buffer
            # normalize_embeddings=True aligns with MPNet's cosine-similarity pre-training
            embeddings = self.model.encode(
                text_buffer,
                batch_size=256,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )

            # Distribute embeddings back to their respective notebooks
            for i, (nb_idx, ctype, cid) in enumerate(meta_buffer):
                emb = embeddings[i]
                if ctype == "code":
                    current_batch_records[nb_idx]["code_emb_map"][cid] = emb
                    current_batch_records[nb_idx]["code_ids_found"].append(cid)
                else:
                    current_batch_records[nb_idx]["md_emb_map"][cid] = emb
                    current_batch_records[nb_idx]["md_ids_found"].append(cid)

            # Reconstruct final rows for this batch
            sorted_indices = sorted(current_batch_records.keys())
            for idx in sorted_indices:
                rec = current_batch_records[idx]

                c_map = rec["code_emb_map"]
                m_map = rec["md_emb_map"]

                # Determine Code Order (Anchors)
                # For Train/Val, we use the ground truth 'cell_order' to ensure anchors are strictly ordered.
                # For Test, we rely on the order of appearance in the JSON (which is preserved by json.load).
                ordered_code_ids = []
                if not is_test and "cell_order" in rec:
                    gt_order = rec["cell_order"].split()
                    ordered_code_ids = [cid for cid in gt_order if cid in c_map]
                else:
                    # Use the order found in the file (preserved in 'code_ids_found')
                    ordered_code_ids = rec["code_ids_found"]

                final_code_embs = [c_map[cid] for cid in ordered_code_ids]

                # Determine Markdown Cells and Labels
                final_md_embs = []
                final_md_ids_list = []
                final_md_labels = []

                # We iterate over the markdown cells found in the file
                for cid in rec["md_ids_found"]:
                    if cid in m_map:
                        final_md_embs.append(m_map[cid])
                        final_md_ids_list.append(cid)

                        if not is_test and "labels" in rec:
                            # Assign ground truth label
                            final_md_labels.append(rec["labels"].get(cid, 0))
                        else:
                            # Dummy label for test
                            final_md_labels.append(-1)

                # Store result
                all_ids.append(rec["id"])
                all_code_embs.append([x.tolist() for x in final_code_embs])
                all_md_embs.append([x.tolist() for x in final_md_embs])
                all_md_labels.append(np.array(final_md_labels, dtype=np.int32))
                all_md_ids.append(final_md_ids_list)

            # Reset buffers
            text_buffer = []
            meta_buffer = []
            current_batch_records = {}
            batch_counter = 0

        # Main Loop over Metadata
        for idx, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]

            nb_json = self._get_notebook_data(filepath)
            if not nb_json:
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Initialize record for this notebook
            rec = {
                "id": nb_id,
                "code_emb_map": {},
                "code_ids_found": [],  # To track file order
                "md_emb_map": {},
                "md_ids_found": [],  # To track file order
            }

            if not is_test:
                rec["cell_order"] = row["cell_order"]
                rec["labels"] = self._generate_labels(row["cell_order"], cell_types)

            current_batch_records[batch_counter] = rec

            # Extract text from cells
            # We iterate over sources. In Python 3.7+, dict insertion order is preserved.
            # This order is reliable for the Test set code anchors.
            for cell_id, text in sources.items():
                ctype = cell_types.get(cell_id, "unknown")

                if ctype not in ["code", "markdown"]:
                    continue

                # Preprocessing: Minimal (strip whitespace)
                text_clean = text.strip()

                text_buffer.append(text_clean)
                meta_buffer.append((batch_counter, ctype, cell_id))

            batch_counter += 1

            # Flush if batch is full
            if batch_counter >= BATCH_SIZE_NBS:
                flush_buffer()

        # Flush any remaining notebooks
        if batch_counter > 0:
            flush_buffer()

        # Create DataFrame
        df_out = pd.DataFrame(
            {
                "id": all_ids,
                "code_embeddings": all_code_embs,
                "markdown_embeddings": all_md_embs,
                "markdown_labels": all_md_labels,
                "markdown_ids": all_md_ids,
            }
        )

        # Save to Parquet
        print(f"Saving features to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_parquet(output_path, engine="pyarrow")

        return df_out

    def run(self):
        """
        Executes the preprocessing pipeline for all splits.
        """
        print("Starting feature extraction pipeline...")

        # Train
        self.process_dataset(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_FEATURES_PATH,
            load_cached_data=True,
            is_test=False,
        )

        # Validation
        self.process_dataset(
            Config.VAL_METADATA_PATH,
            Config.VAL_FEATURES_PATH,
            load_cached_data=True,
            is_test=False,
        )

        # Test
        self.process_dataset(
            Config.TEST_METADATA_PATH,
            Config.TEST_FEATURES_PATH,
            load_cached_data=True,
            is_test=True,
        )
        print("Feature extraction completed.")
