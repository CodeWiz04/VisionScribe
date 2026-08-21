from pathlib import Path
from collections import Counter

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"
rows = []
def load_captions(captions_file:Path)->pd.DataFrame:
    with captions_file.open("r",encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            image_name, caption = line.split(",", maxsplit=1)  #will split at first comma

            rows.append(
                {
                    "image": image_name,
                    "caption": caption,
                }
            )

    return pd.DataFrame(rows)
    