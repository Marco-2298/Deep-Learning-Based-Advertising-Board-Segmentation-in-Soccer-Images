from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT / "data" / "processed" / "rf_detr"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "figures"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualizza un campione del dataset COCO."
    )

    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="train",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    split_directory = DATASET_DIR / arguments.split
    annotations_path = (
        split_directory / "_annotations.coco.json"
    )

    with annotations_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        coco_data = json.load(file)

    images_by_id = {
        image["id"]: image
        for image in coco_data["images"]
    }

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in coco_data["annotations"]:
        annotations_by_image[
            annotation["image_id"]
        ].append(annotation)

    positive_ids = [
        image_id
        for image_id in images_by_id
        if len(annotations_by_image[image_id]) > 0
    ]

    zero_annotation_ids = [
        image_id
        for image_id in images_by_id
        if len(annotations_by_image[image_id]) == 0
    ]

    random_generator = random.Random(arguments.seed)

    number_of_zero_annotation_samples = min(
        2,
        len(zero_annotation_ids),
        arguments.samples,
    )

    number_of_positive_samples = (
        arguments.samples - number_of_zero_annotation_samples
    )

    selected_ids = random_generator.sample(
        positive_ids,
        min(number_of_positive_samples, len(positive_ids)),
    )

    if number_of_zero_annotation_samples > 0:
        selected_ids.extend(
            random_generator.sample(
                zero_annotation_ids,
                number_of_zero_annotation_samples,
            )
        )

    random_generator.shuffle(selected_ids)

    columns = 3
    rows = (len(selected_ids) + columns - 1) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 5 * rows),
    )

    axes = axes.flatten()

    for axis, image_id in zip(axes, selected_ids):
        image_data = images_by_id[image_id]

        image_path = (
            split_directory / image_data["file_name"]
        )

        image = Image.open(image_path).convert("RGB")

        annotations = annotations_by_image[image_id]

        axis.imshow(image)

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]

            rectangle = Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                linewidth=2,
            )

            axis.add_patch(rectangle)

            axis.text(
                x,
                max(0, y - 4),
                "advertising_board",
                fontsize=7,
                bbox={
                    "alpha": 0.7,
                },
            )

        axis.set_title(
            f"{image_data['file_name']}\n"
            f"{len(annotations)} bounding box",
            fontsize=9,
        )

        axis.axis("off")

    for axis in axes[len(selected_ids):]:
        axis.axis("off")

    figure.suptitle(
        f"COCO dataset — split {arguments.split}",
        fontsize=16,
    )

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"coco_{arguments.split}_samples.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Figura salvata in: {output_path}")
    print(
        f"Campione: {len(selected_ids)} immagini, "
        f"di cui {number_of_zero_annotation_samples} "
        "senza annotazioni"
    )


if __name__ == "__main__":
    main()