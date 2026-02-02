import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config


class Preprocessor:
    def __init__(self):
        self.config = Config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_seed(self):
        seed = self.config.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_json(self, filepath):
        full_path = os.path.join(self.config.INPUT_DIR, filepath)
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_notebook_data(self, row, is_test=False):
        """
        Extracts code and markdown text/ids from a notebook.
        For train/val, calculates ground truth ranks (labels).
        """
        nb_id = row["id"]
        filepath = row["filepath"]

        try:
            data = self.load_json(filepath)
        except FileNotFoundError:
            return None

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        code_ids = []
        md_ids = []
        md_labels = []

        if not is_test:
            # Use ground truth order to determine structure and labels
            if isinstance(row["cell_order"], str):
                cell_order = row["cell_order"].split()
            else:
                return None

            # Determine labels: Label = number of code cells appearing before the markdown cell
            current_code_count = 0

            # Temporary storage to ensure alignment
            temp_md_data = []  # list of (id, label)

            for cell_id in cell_order:
                ctype = cell_types.get(cell_id)
                if ctype == "code":
                    code_ids.append(cell_id)
                    current_code_count += 1
                elif ctype == "markdown":
                    temp_md_data.append((cell_id, current_code_count))

            # Unpack markdown data
            for mid, mlabel in temp_md_data:
                md_ids.append(mid)
                md_labels.append(mlabel)

        else:
            # For test set, we rely on the order in the JSON file.
            # We filter keys by type.
            for cell_id, ctype in cell_types.items():
                if ctype == "code":
                    code_ids.append(cell_id)
                elif ctype == "markdown":
                    md_ids.append(cell_id)

            # No labels for test
            md_labels = [-1] * len(md_ids)

        # Extract text content
        code_texts = [sources.get(cid, "") for cid in code_ids]
        md_texts = [sources.get(mid, "") for mid in md_ids]

        return {
            "id": nb_id,
            "code_ids": code_ids,
            "md_ids": md_ids,
            "code_texts": code_texts,
            "md_texts": md_texts,
            "md_labels": md_labels,
        }

    def process_dataset(self, metadata_path, output_path, is_test=False):
        """
        Loads metadata, processes notebooks, generates embeddings, and saves to Parquet.
        """
        print(f"Processing {metadata_path}...")

        # Load metadata
        df_meta = pd.read_csv(metadata_path)

        if self.config.DEBUG:
            df_meta = df_meta.head(self.config.DEBUG_SAMPLES)
            print(f"Debug mode: Processing {len(df_meta)} samples.")

        # Load Model
        print(f"Loading model {self.config.MODEL_NAME}...")
        model = SentenceTransformer(self.config.MODEL_NAME, device=self.device)

        # Processing loop with batching
        batch_size = 256  # Number of notebooks to process in memory before encoding

        records = []
        batch_buffer = []

        # Helper to flush buffer
        def flush_batch(buffer):
            if not buffer:
                return []

            # Flatten texts for bulk encoding
            all_code_texts = []
            all_md_texts = []

            # Track counts to reconstruct structure later
            notebook_meta = []

            for item in buffer:
                c_txt = item["code_texts"]
                m_txt = item["md_texts"]

                notebook_meta.append(
                    {
                        "id": item["id"],
                        "code_ids": item["code_ids"],
                        "md_ids": item["md_ids"],
                        "md_labels": item["md_labels"],
                        "n_code": len(c_txt),
                        "n_md": len(m_txt),
                    }
                )
                all_code_texts.extend(c_txt)
                all_md_texts.extend(m_txt)

            # Encode Code
            code_embeddings = []
            if all_code_texts:
                code_embeddings = model.encode(
                    all_code_texts,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device,
                )

            # Encode Markdown
            md_embeddings = []
            if all_md_texts:
                md_embeddings = model.encode(
                    all_md_texts,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device,
                )

            # Reconstruct notebook records
            processed_records = []
            c_ptr = 0
            m_ptr = 0

            for meta in notebook_meta:
                n_c = meta["n_code"]
                n_m = meta["n_md"]

                c_emb = (
                    code_embeddings[c_ptr : c_ptr + n_c] if n_c > 0 else np.array([])
                )
                m_emb = md_embeddings[m_ptr : m_ptr + n_m] if n_m > 0 else np.array([])

                # Convert to list for Parquet compatibility
                c_emb_list = c_emb.tolist() if len(c_emb) > 0 else []
                m_emb_list = m_emb.tolist() if len(m_emb) > 0 else []

                record = {
                    "id": meta["id"],
                    "code_embeddings": c_emb_list,
                    "markdown_embeddings": m_emb_list,
                    "code_ids": meta["code_ids"],
                    "markdown_ids": meta["md_ids"],
                    "markdown_labels": meta["md_labels"],
                }
                processed_records.append(record)

                c_ptr += n_c
                m_ptr += n_m

            return processed_records

        # Main Loop
        total = len(df_meta)
        for idx, row in df_meta.iterrows():
            data = self.get_notebook_data(row, is_test)
            if data:
                batch_buffer.append(data)

            if len(batch_buffer) >= batch_size:
                records.extend(flush_batch(batch_buffer))
                batch_buffer = []
                if idx % 1000 == 0:
                    print(f"Processed {idx}/{total} notebooks...", end="\r")

        # Flush remaining
        if batch_buffer:
            records.extend(flush_batch(batch_buffer))

        print(f"Finished processing {len(records)} notebooks.")

        # Save to Parquet
        print(f"Saving to {output_path}...")
        df_out = pd.DataFrame(records)
        df_out.to_parquet(output_path, index=False)
        print("Save complete.")

    def generate_embeddings(self, load_cached_data=True):
        self.set_seed()

        # Define tasks: (Metadata Path, Output Path, Is Test)
        tasks = [
            (self.config.TRAIN_METADATA_PATH, self.config.TRAIN_FEATURES_PATH, False),
            (self.config.VAL_METADATA_PATH, self.config.VAL_FEATURES_PATH, False),
            (self.config.TEST_METADATA_PATH, self.config.TEST_FEATURES_PATH, True),
        ]

        for meta_path, out_path, is_test in tasks:
            if load_cached_data and os.path.exists(out_path):
                print(f"Found cached features at {out_path}. Skipping generation.")
                continue

            if not os.path.exists(meta_path):
                print(f"Metadata file {meta_path} not found. Skipping.")
                continue

            self.process_dataset(meta_path, out_path, is_test)


def generate_embeddings(load_cached_data=True):
    """
    Wrapper function to be imported by other modules.
    """
    preprocessor = Preprocessor()
    preprocessor.generate_embeddings(load_cached_data=load_cached_data)
