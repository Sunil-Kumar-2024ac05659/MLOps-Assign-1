"""
Downloads the Heart Disease UCI Dataset (id=45) and saves it to data/raw/.
Usage: python data/download_data.py
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_dataset(output_dir: str = "data/raw") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "heart_disease.csv")

    print("Fetching Heart Disease UCI Dataset (id=45)...")
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features + target; binarise target (0 = no disease, 1 = disease)
    df = pd.concat([X, y], axis=1)
    df.rename(columns={"num": "target"}, inplace=True)
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}  ({len(df)} rows, {len(df.columns)} columns)")
    return output_path


if __name__ == "__main__":
    download_dataset()
