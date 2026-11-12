
import json
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

# Reproducible training
tf.keras.utils.set_random_seed(SEED)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# ============================================================
# 2. VALIDATE DATASET
# ============================================================

print("=" * 60)
print("          DeepLeaf AI - CNN Training")
print("=" * 60)

for split in ["train", "validation", "test"]:
    split_path = DATA_DIR / split

    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {split_path}"
        )

train_classes = sorted(
    p.name for p in (DATA_DIR / "train").iterdir()
    if p.is_dir()
)

validation_classes = sorted(
    p.name for p in (DATA_DIR / "validation").iterdir()
    if p.is_dir()
)

test_classes = sorted(
    p.name for p in (DATA_DIR / "test").iterdir()
    if p.is_dir()
)

if not train_classes:
    raise ValueError("No class folders found in the training dataset.")

if train_classes != validation_classes or train_classes != test_classes:
    raise ValueError(
        "Class folders must match in train, validation, and test."
    )

class_names = train_classes
num_classes = len(class_names)

print(f"\nNumber of disease classes: {num_classes}")

for name in class_names:
    print(f"  - {name}")

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# 3. COUNT TRAINING IMAGES
# ============================================================

print("\nCounting training images...")

class_counts = Counter()

for class_name in class_names:
    class_dir = DATA_DIR / "train" / class_name

    count = sum(
        1 for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if count == 0:
        raise ValueError(
            f"No images found in training class: {class_name}"
        )

    class_counts[class_name] = count
    print(f"{class_name}: {count} images")

total_images = sum(class_counts.values())

print(f"\nTotal training images: {total_images}")


# ============================================================
# 4. LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)


# ============================================================
# 5. OPTIMIZE DATA PIPELINE
# ============================================================

# Prefetch one batch to limit extra memory usage on an 8 GB laptop.
train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)
test_ds = test_ds.prefetch(1)


# ============================================================
# 6. CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

# Calculate balanced weights using folder image counts.
class_weights = {
    class_id: total_images / (num_classes * class_counts[class_name])
    for class_id, class_name in enumerate(class_names)
}

for class_id, weight in class_weights.items():
    print(f"{class_names[class_id]}: {weight:.3f}")


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# 8. BUILD CNN MODEL
# ============================================================

print("\nBuilding CNN model...")

model = keras.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # Normalize RGB pixels from 0-255 to 0-1
        layers.Rescaling(1.0 / 255),

        data_augmentation,

        # CNN Block 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # CNN Block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # CNN Block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # CNN Block 4
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Feature extraction
        layers.GlobalAveragePooling2D(),

        # Classification layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        # One output for each disease class
        layers.Dense(num_classes, activation="softmax")
    ],
    name="DeepLeaf_CNN"
)

model.summary()


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. CALLBACKS
# ============================================================

checkpoint_path = MODEL_DIR / "deepleaf_cnn_best.keras"

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=1,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nStarting CNN training...")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Training images: {total_images}")
print("\nPlease keep your laptop plugged in.\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

print("\nLoading the best checkpoint...")

if not checkpoint_path.exists():
    raise FileNotFoundError(
        "Best model checkpoint was not created."
    )

best_model = keras.models.load_model(checkpoint_path)

final_model_path = MODEL_DIR / "deepleaf_cnn_final.keras"
best_model.save(final_model_path)

print(f"Best model saved to: {final_model_path}")


# ============================================================
# 13. EVALUATE ON TEST DATA
# ============================================================

print("\nEvaluating on the test dataset...")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy * 100:.2f}%")
print("=" * 60)


# ============================================================
# 14. SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

history_path = MODEL_DIR / "training_history.json"

with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history_data, f, indent=4)

print(f"\nTraining history saved to: {history_path}")
print("\nDeepLeaf AI training completed successfully!")
