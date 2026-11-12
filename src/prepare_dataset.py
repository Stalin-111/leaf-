import os
import shutil
import random
from pathlib import Path

SOURCE = Path("dataset/raw")
DEST = Path("dataset/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    return [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]


def main():

    if not SOURCE.exists():
        print(f"Dataset not found: {SOURCE}")
        return

    classes = sorted([
        folder for folder in SOURCE.iterdir()
        if folder.is_dir()
    ])

    if not classes:
        print("No class folders found.")
        return

    print("=" * 60)
    print("        DeepLeaf AI - Dataset Preparation")
    print("=" * 60)

    print(f"\nClasses found: {len(classes)}")

    # Create destination folders
    for split in ["train", "validation", "test"]:
        (DEST / split).mkdir(parents=True, exist_ok=True)

    total = 0

    for class_folder in classes:

        images = get_images(class_folder)

        random.shuffle(images)

        n = len(images)

        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        splits = {
            "train": train_images,
            "validation": val_images,
            "test": test_images
        }

        print(f"\n{class_folder.name}")
        print(f"  Total      : {n}")
        print(f"  Train      : {len(train_images)}")
        print(f"  Validation : {len(val_images)}")
        print(f"  Test       : {len(test_images)}")

        for split, split_images in splits.items():

            destination = DEST / split / class_folder.name
            destination.mkdir(parents=True, exist_ok=True)

            for image in split_images:

                target = destination / image.name

                # Avoid accidental duplicate filename overwrite
                if target.exists():
                    target = destination / f"{image.stem}_{random.randint(1000, 999999)}{image.suffix}"

                shutil.copy2(image, target)

        total += n

    print("\n" + "=" * 60)
    print(f"Total images processed: {total}")
    print("Dataset preparation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()