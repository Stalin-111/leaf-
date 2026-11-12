import os

DATASET_PATH = "dataset/raw"

if not os.path.exists(DATASET_PATH):
    print(f"Dataset folder not found: {DATASET_PATH}")
    print("Make sure your dataset is inside dataset/raw/")
    exit()

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("=" * 50)
print("       DeepLeaf AI - Dataset Analysis")
print("=" * 50)

print(f"\nNumber of classes: {len(classes)}\n")

total_images = 0

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{class_name}: {len(images)} images")
    total_images += len(images)

print("\n" + "=" * 50)
print(f"Total images: {total_images}")
print("=" * 50)