import pandas as pd
import sqlite3
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
import torch
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


DATABASE_PATH = "../data/data_drifting/database.db"
DATA_ROOT = Path("../data/raw/lfw-deepfunneled")
REFERENCE_DATA_PATH = Path("../data/data_drifting/reference_data.csv")


def calculate_image_properties(image: Image.Image) -> dict:
    """Calculate basic properties of the image.
        Args:
          image: PIL Image object"""
    stat = ImageStat.Stat(image)
    brightness = np.mean(stat.mean)
    contrast = np.mean(stat.stddev)

    return {
        "brightness": brightness,
        "contrast": contrast
    }

def create_reference_dataframe() -> pd.DataFrame:
    """Creates the reference dataset from train split with variables needed for data drift detection."""

    rows = []

    for person_path in DATA_ROOT.iterdir():
        for img_path in sorted(person_path.glob("*.jpg")):
            img = Image.open(img_path)
            properties = calculate_image_properties(img)
            rows.append(
                {
                    "brightness": properties["brightness"],
                    "contrast": properties["contrast"],
                }
            )
    return pd.DataFrame(rows)


if REFERENCE_DATA_PATH.exists():
    print(f"Loading existing reference data from {REFERENCE_DATA_PATH}")
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
else:
    print("Reference data not found. Creating new reference dataset...")
    reference_df = create_reference_dataframe()
    reference_df.dropna(inplace=True)
    reference_df.to_csv(REFERENCE_DATA_PATH, index=False)

pred_df = pd.read_sql_query("SELECT * FROM predictions", sqlite3.connect(DATABASE_PATH))
pred_df.dropna(inplace=True)

# Keep only shared columns to avoid NaNs/shape issues
shared_cols = [c for c in reference_df.columns if c in pred_df.columns]
if not shared_cols:
    raise ValueError("No shared columns between reference and prediction dataframes.")
reference_df = reference_df[shared_cols]
pred_df = pred_df[shared_cols]

print(pred_df.head())
print(reference_df.head())

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=pred_df)
report.save_html("report.html")

