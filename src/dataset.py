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

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    train_df.to_csv(
        PROCESSED_DIR / "train_captions.csv",
        index=False,
    )

    val_df.to_csv(
        PROCESSED_DIR / "val_captions.csv",
        index=False,
    )

    test_df.to_csv(
        PROCESSED_DIR / "test_captions.csv",
        index=False,
    )

    print("\nSplit files saved to:")
    print(PROCESSED_DIR)

def inspect_samples(captions_df:pd.DataFrame,num_images:int=5) -> None:
    
    


def main() -> None:
    """
    Run the complete Flickr8k dataset preparation process.
    """

    print("Loading Flickr8k captions...")

    captions_df = load_captions(CAPTIONS_FILE)

    print(f"Total captions loaded: {len(captions_df)}")
    print(
        f"Unique images: "
        f"{captions_df['image'].nunique()}"
    )

    # Inspect a few examples before splitting.
    inspect_samples(captions_df)

    # Split by image.
    train_df, val_df, test_df = split_images(captions_df)

    # Verify there is no image leakage.
    verify_split(
        train_df,
        val_df,
        test_df,
    )

    # Save metadata.
    save_splits(
        train_df,
        val_df,
        test_df,
    )
if __name__ == "__main__":
    main()
    