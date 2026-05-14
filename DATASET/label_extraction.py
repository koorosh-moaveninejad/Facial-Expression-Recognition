import os

import csv

import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_MAP = {
    "angry": 1,
    "contempt": 2,
    "disgust": 3,
    "fear": 4,
    "happy": 5,
    "neutral": 6,
    "sad": 7,
    "suprise": 8
}

def create_csv(dataset_root, split_name, output_csv):

    split_path = os.path.join(dataset_root, split_name)

    rows = []

    for folder_name in sorted(os.listdir(split_path)):

        folder_path = os.path.join(split_path, folder_name)

        if not os.path.isdir(folder_path):

            continue

        key = folder_name.lower().strip()

        if key not in LABEL_MAP:

            raise ValueError(f"Unknown expression folder: {folder_name}")

        label = LABEL_MAP[key]

        for image_name in sorted(os.listdir(folder_path)):

            ext = os.path.splitext(image_name)[1].lower()

            if ext not in IMAGE_EXTENSIONS:

                continue

            rows.append({

                "image": image_name,

                "label": label,

                "folder": folder_name

            })

    with open(output_csv, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=["image", "label", "folder"])

        writer.writeheader()

        writer.writerows(rows)

    print(f"Created {output_csv}: {len(rows)} images")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default=".")

    args = parser.parse_args()

    create_csv(args.dataset_root, "train", os.path.join(args.dataset_root, "train_labels.csv"))

    create_csv(args.dataset_root, "validation", os.path.join(args.dataset_root, "validation_labels.csv"))

    create_csv(args.dataset_root, "test", os.path.join(args.dataset_root, "test_labels.csv"))