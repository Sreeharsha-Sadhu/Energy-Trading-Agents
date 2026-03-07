# src/data_generation/writer.py
import os
from typing import List, Dict

import pandas as pd


def write_data_chunk(
        data_chunk: List[Dict], output_filename: str, is_first_chunk: bool = False
) -> None:
    """
    Write a chunk of records to CSV. Creates parent dir if needed.
    """
    if not data_chunk:
        return
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    df_chunk = pd.DataFrame(data_chunk)
    if is_first_chunk:
        df_chunk.to_csv(output_filename, index=False, mode="w")
    else:
        df_chunk.to_csv(output_filename, index=False, mode="a", header=False)
