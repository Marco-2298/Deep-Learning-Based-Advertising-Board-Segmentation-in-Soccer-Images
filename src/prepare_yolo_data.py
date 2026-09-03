from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "rf_detr"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "yolo"
)

SPLITS = ("train", "valid", "test")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Converte il dataset COCO già preparato per RF-DETR "
            "nel formato YOLO mantenendo esattamente gli stessi split."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory contenente gli split COCO RF-DETR.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory nella quale creare il dataset YOLO.",
    )

    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help=(
            "Metodo per trasferire le immagini. "
            "'hardlink' evita di duplicare lo spazio occupato."
        ),
    )

    return parser.parse_args()


def reset_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def transfer_image(
    source: Path,
    destination: Path,
    copy_mode: str,
) -> None:
    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Converte:

        COCO: [x_min, y_min, width, height]

    in:

        YOLO:
        [x_center, y_center, width, height]

    con coordinate normalizzate tra 0 e 1.
    """
    x_min, y_min, width, height = bbox

    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    x_center /= image_width
    y_center /= image_height

    normalized_width = width / image_width
    normalized_height = height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )


def validate_yolo_bbox(
    bbox: tuple[float, float, float, float],
) -> bool:
    x_center, y_center, width, height = bbox

    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )


def convert_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    copy_mode: str,
) -> dict[str, Any]:

    source_split_dir = input_dir / split_name

    annotations_path = (
        source_split_dir / "_annotations.coco.json"
    )

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotazioni COCO non trovate: {annotations_path}"
        )

    images_output_dir = (
        output_dir / "images" / split_name
    )

    labels_output_dir = (
        output_dir / "labels" / split_name
    )

    images_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with annotations_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        coco_data = json.load(file)

    images = coco_data["images"]
    annotations = coco_data["annotations"]

    annotations_by_image: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for annotation in annotations:
        annotations_by_image[
            annotation["image_id"]
        ].append(annotation)

    total_labels = 0
    zero_annotation_images = 0
    invalid_boxes = 0

    for index, image_info in enumerate(
        images,
        start=1,
    ):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        source_image = source_split_dir / file_name

        if not source_image.exists():
            raise FileNotFoundError(
                f"Immagine non trovata: {source_image}"
            )

        destination_image = (
            images_output_dir / file_name
        )

        transfer_image(
            source=source_image,
            destination=destination_image,
            copy_mode=copy_mode,
        )

        image_annotations = annotations_by_image.get(
            image_id,
            [],
        )

        if len(image_annotations) == 0:
            zero_annotation_images += 1

        label_path = (
            labels_output_dir
            / f"{Path(file_name).stem}.txt"
        )

        label_lines: list[str] = []

        for annotation in image_annotations:
            # Il dataset RF-DETR usa category_id = 1,
            # ma YOLO utilizza classi zero-based.
            class_id = 0

            yolo_bbox = coco_bbox_to_yolo(
                bbox=annotation["bbox"],
                image_width=image_width,
                image_height=image_height,
            )

            if not validate_yolo_bbox(yolo_bbox):
                invalid_boxes += 1
                continue

            (
                x_center,
                y_center,
                width,
                height,
            ) = yolo_bbox

            label_lines.append(
                (
                    f"{class_id} "
                    f"{x_center:.8f} "
                    f"{y_center:.8f} "
                    f"{width:.8f} "
                    f"{height:.8f}"
                )
            )

            total_labels += 1

        with label_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            file.write("\n".join(label_lines))

            if label_lines:
                file.write("\n")

        if (
            index % 500 == 0
            or index == len(images)
        ):
            print(
                f"[{split_name}] "
                f"{index}/{len(images)} immagini convertite"
            )

    return {
        "images": len(images),
        "labels": total_labels,
        "zero_annotation_images": zero_annotation_images,
        "invalid_boxes": invalid_boxes,
    }


def write_dataset_yaml(
    output_dir: Path,
) -> Path:

    yaml_path = output_dir / "dataset.yaml"

    dataset_configuration = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {
            0: "advertising_board",
        },
    }

    with yaml_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            dataset_configuration,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return yaml_path


def main() -> None:
    arguments = parse_arguments()

    input_dir = arguments.input_dir.resolve()
    output_dir = arguments.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Dataset RF-DETR non trovato: {input_dir}"
        )

    print("\nPreparazione dataset YOLO")
    print("=========================")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    reset_directory(output_dir)

    summaries: dict[str, Any] = {}

    for split_name in SPLITS:
        print(
            f"\nConversione split: {split_name}"
        )

        summaries[split_name] = convert_split(
            split_name=split_name,
            input_dir=input_dir,
            output_dir=output_dir,
            copy_mode=arguments.copy_mode,
        )

    yaml_path = write_dataset_yaml(
        output_dir=output_dir,
    )

    summary_path = (
        output_dir / "conversion_summary.json"
    )

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summaries,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nConversione completata")
    print("======================")

    total_images = 0
    total_labels = 0

    for split_name, summary in summaries.items():
        total_images += summary["images"]
        total_labels += summary["labels"]

        print(
            f"{split_name}: "
            f"{summary['images']} immagini, "
            f"{summary['labels']} bounding box, "
            f"{summary['zero_annotation_images']} "
            f"immagini senza annotazioni"
        )

    print(f"\nTotale immagini: {total_images}")
    print(f"Totale bounding box: {total_labels}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()