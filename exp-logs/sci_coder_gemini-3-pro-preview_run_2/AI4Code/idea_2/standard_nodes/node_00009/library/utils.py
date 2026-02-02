import os
import json
from library.config import INPUT_DIR, seed_everything


def read_notebook(filepath):
    """
    Safely loads and parses a JSON notebook file.

    Args:
        filepath (str): The relative path to the notebook file (e.g., 'train/{id}.json').
                        This path is appended to the global INPUT_DIR.

    Returns:
        dict: The parsed JSON content of the notebook. Returns an empty dictionary if
              the file cannot be found or parsed.
    """
    full_path = os.path.join(INPUT_DIR, filepath)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        # Return an empty dict to allow the pipeline to continue gracefully
        # In a verbose setting, we might log: print(f"Error reading {full_path}: {e}")
        return {}
