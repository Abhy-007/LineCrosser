import os
import random
import shutil

image_dir = "data/images/all"
label_dir = "data/labels"

train_img_dir = "data/images/train"
val_img_dir = "data/images/val"

train_lbl_dir = "data/labels/train"
val_lbl_dir = "data/labels/val"

split_ratio = 0.8  # 80% train

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

valid_images = []
skipped_images = []

# Check if label exists
for img in images:
    label = img.replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, label)

    if os.path.exists(label_path):
        valid_images.append(img)
    else:
        skipped_images.append(img)

print(f"Total images found: {len(images)}")
print(f"Valid image-label pairs: {len(valid_images)}")
print(f"Skipped images (missing labels): {len(skipped_images)}")

for img in skipped_images:
    print("Skipped:", img)

# Shuffle valid images
random.shuffle(valid_images)

split_index = int(len(valid_images) * split_ratio)

train_images = valid_images[:split_index]
val_images = valid_images[split_index:]


def move_files(image_list, img_dest, lbl_dest):
    for img in image_list:
        label = img.replace(".jpg", ".txt")

        shutil.move(
            os.path.join(image_dir, img),
            os.path.join(img_dest, img)
        )

        shutil.move(
            os.path.join(label_dir, label),
            os.path.join(lbl_dest, label)
        )


move_files(train_images, train_img_dir, train_lbl_dir)
move_files(val_images, val_img_dir, val_lbl_dir)

print("\nDataset split complete.")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")