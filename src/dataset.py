from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"
RANDOM_SEED=42
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

def split_images(
    captions_df:pd.DataFrame,
    random_seed:int=RANDOM_SEED,
)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_images = captions_df["image"].unique()
    train_images, temp_images = train_test_split(
        unique_images,
        test_size=0.20,
        random_state=random_seed,
    )
    val_images,test_images=train_test_split(
        temp_images,
        test_size=0.5,
        random_state=random_seed
    )
    train_df=captions_df[
        captions_df["image"].isin(train_images)
    ].copy()
    val_df = captions_df[
        captions_df["image"].isin(val_images)
    ].copy()

    test_df = captions_df[
        captions_df["image"].isin(test_images)
    ].copy()

    return train_df, val_df, test_df
    
def verify_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_images = set(train_df["image"])
    val_images = set(val_df["image"])
    test_images = set(test_df["image"])
    print("\n========== SPLIT VERIFICATION ==========")

    print(f"Training images:   {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images:       {len(test_images)}")

    print(f"\nTraining captions:   {len(train_df)}")
    print(f"Validation captions: {len(val_df)}")
    print(f"Test captions:       {len(test_df)}")
    print("\nImage overlap:")

    print(
        f"Train ∩ Validation: "
        f"{len(train_images & val_images)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_images & test_images)}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(val_images & test_images)}"
    )
    assert not train_images.intersection(val_images)
    assert not train_images.intersection(test_images)
    assert not val_images.intersection(test_images)

    print("\n✓ No image appears in more than one split.")

def inspect_dataset() -> None:
    """
    Inspect the Flickr8k dataset and print basic statistics.
    """
    print("Loading Flickr8k captions...")

    captions_df = load_captions(CAPTIONS_FILE)

    image_files = list(IMAGE_DIR.glob("*.jpg"))
    image_names = {image.name for image in image_files}

    caption_image_names = set(captions_df["image"])

    caption_counts = Counter(captions_df["image"])

    print("\n========== DATASET STATISTICS ==========")

    print(f"Image files found: {len(image_files)}")
    print(f"Unique images in captions.txt: {len(caption_image_names)}")
    print(f"Total captions: {len(captions_df)}")

    print(
        f"Average captions per image: "
        f"{len(captions_df) / len(caption_image_names):.2f}"
    )

    print(
        f"Images with exactly 5 captions: "
        f"{sum(count == 5 for count in caption_counts.values())}"
    )

    missing_images = caption_image_names - image_names

    print(f"Caption entries with missing image files: {len(missing_images)}")

    print("\nCaption distribution:")
    print(Counter(caption_counts.values()))

    print("\n========== SAMPLE CAPTIONS ==========")

    print(captions_df.head(10).to_string(index=False))


if __name__ == "__main__":
    inspect_dataset()
    