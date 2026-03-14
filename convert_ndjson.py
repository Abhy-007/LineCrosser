import json
import os

ndjson_path = "/home/theabman/Downloads/test.ndjson"   # your downloaded NDJSON file
image_dir = "/home/theabman/Desktop/PERSONAL PROJECTS/LineCrosser/data/images/all"
label_dir = "/home/theabman/Desktop/PERSONAL PROJECTS/LineCrosser/data/labels"

os.makedirs(label_dir, exist_ok=True)

class_names = {}

with open(ndjson_path, "r") as f:
    for line in f:
        item = json.loads(line)

        # read class names
        if item["type"] == "dataset":
            class_names = item["class_names"]

        # read annotations
        if item["type"] == "image":
            file_name = item["file"]
            boxes = item["annotations"]["boxes"]

            label_path = os.path.join(label_dir, file_name.replace(".jpg", ".txt"))

            with open(label_path, "w") as lf:
                for box in boxes:
                    cls, x, y, w, h = box
                    lf.write(f"{cls} {x} {y} {w} {h}\n")

print("Conversion complete.")
print("Classes:", class_names)