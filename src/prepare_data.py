from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils import (
    bbox_xyxy_to_coco,
    get_object_bbox,
    is_valid_bbox,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGES_DIR = (
    PROJECT_ROOT / "data" / "raw" / "detection" / "images"
)

DEFAULT_ANNOTATIONS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "detection" / "annotations"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "rf_detr"
)

# Le sei classi sponsor originali vengono accorpate perché
# il task richiede solamente la localizzazione del pannello.

ALLOWED_CLASSES = {
    "heineken",
    "mastercard",
    "gazprom",
    "playstation",
    "nissan",
    "pepsi",
}

COCO_CATEGORY = {
    "id": 1,
    "name": "advertising_board",
    "supercategory": "advertising_board",
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Converte il dataset Supervisely dei cartelloni "
            "pubblicitari nel formato COCO richiesto da RF-DETR."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Cartella contenente le immagini originali.",
    )

    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Cartella contenente i file JSON Supervisely.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella nella quale creare train, valid e test.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Percentuale di immagini per il training set.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Percentuale di immagini per il validation set.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per rendere riproducibile lo split.",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        # Esclude le annotazioni residuali estremamente piccole
        # individuate durante l'analisi preliminare del dataset.
        help="Area minima in pixel per mantenere una bounding box.",
    )

    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help=(
            "Metodo utilizzato per trasferire le immagini. "
            "'hardlink' evita di duplicare lo spazio occupato."
        ),
    )

    return parser.parse_args()


def find_dataset_pairs(
    images_dir: Path,
    annotations_dir: Path,
) -> tuple[
    list[tuple[Path, Path]],
    list[Path],
    list[Path],
]:
    """
    Associa ogni immagine al relativo JSON Supervisely.

    Un file come:

        example.png.json

    viene associato a:

        example.png
    """
    image_paths = {
        path.name: path
        for path in images_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    }

    annotation_paths = {
        path.name.removesuffix(".json"): path
        for path in annotations_dir.glob("*.json")
        if path.is_file()
    }

    valid_names = sorted(
        set(image_paths).intersection(annotation_paths)
    )

    valid_pairs = [
        (image_paths[name], annotation_paths[name])
        for name in valid_names
    ]

    images_without_annotations = [
        image_paths[name]
        for name in sorted(
            set(image_paths).difference(annotation_paths)
        )
    ]

    orphan_annotations = [
        annotation_paths[name]
        for name in sorted(
            set(annotation_paths).difference(image_paths)
        )
    ]

    return (
        valid_pairs,
        images_without_annotations,
        orphan_annotations,
    )


def split_dataset(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    if train_ratio <= 0 or valid_ratio <= 0:
        raise ValueError(
            "train_ratio e valid_ratio devono essere positivi."
        )

    if train_ratio + valid_ratio >= 1:
        raise ValueError(
            "La somma di train_ratio e valid_ratio deve essere "
            "inferiore a 1."
        )

    shuffled_pairs = pairs.copy()

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_pairs)

    total_images = len(shuffled_pairs)

    train_size = round(total_images * train_ratio)
    valid_size = round(total_images * valid_ratio)

    train_end = train_size
    valid_end = train_size + valid_size

    return {
        "train": shuffled_pairs[:train_end],
        "valid": shuffled_pairs[train_end:valid_end],
        "test": shuffled_pairs[valid_end:],
    }


def reset_split_directory(split_directory: Path) -> None:
    """
    Elimina un eventuale split precedente e lo ricrea vuoto.
    """
    if split_directory.exists():
        shutil.rmtree(split_directory)

    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def transfer_image(
    source: Path,
    destination: Path,
    copy_mode: str,
) -> None:
    """
    Copia oppure crea un hard link dell'immagine.

    In caso di errore nella creazione dell'hard link viene
    utilizzata automaticamente una copia tradizionale.
    """
    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_coco_split(
    split_name: str,
    pairs: list[tuple[Path, Path]],
    split_directory: Path,
    min_area: int,
    copy_mode: str,
) -> dict[str, Any]:
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []

    source_class_counts: Counter[str] = Counter()

    skipped_empty_masks = 0
    skipped_small_boxes = 0
    skipped_invalid_boxes = 0
    skipped_unknown_classes = 0
    size_mismatches = 0
    zero_annotation_images = 0

    annotation_id = 1

    total_split_images = len(pairs)

    for image_id, (image_path, annotation_path) in enumerate(
        pairs,
        start=1,
    ):
        destination_path = split_directory / image_path.name

        transfer_image(
            source=image_path,
            destination=destination_path,
            copy_mode=copy_mode,
        )

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except Exception as error:
            raise RuntimeError(
                f"Impossibile leggere l'immagine {image_path}"
            ) from error

        try:
            with annotation_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                supervisely_data = json.load(file)
        except Exception as error:
            raise RuntimeError(
                f"Impossibile leggere {annotation_path}"
            ) from error

        json_width = supervisely_data["size"]["width"]
        json_height = supervisely_data["size"]["height"]

        if (
            image_width != json_width
            or image_height != json_height
        ):
            size_mismatches += 1

        objects = supervisely_data.get("objects", [])

        if len(objects) == 0:
            zero_annotation_images += 1

        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        for obj in objects:
            source_class = str(
                obj.get("classTitle", "")
            ).lower()

            source_class_counts[source_class] += 1

            if source_class not in ALLOWED_CLASSES:
                skipped_unknown_classes += 1
                continue

            bbox_xyxy = get_object_bbox(obj)

            if bbox_xyxy is None:
                skipped_empty_masks += 1
                continue

            bbox_coco = bbox_xyxy_to_coco(bbox_xyxy)

            x_min, y_min, width, height = bbox_coco
            area = width * height

            if area < min_area:
                skipped_small_boxes += 1
                continue

            if not is_valid_bbox(
                bbox=bbox_xyxy,
                image_width=image_width,
                image_height=image_height,
                min_area=min_area,
            ):
                skipped_invalid_boxes += 1
                continue

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY["id"],
                    "bbox": [
                        int(x_min),
                        int(y_min),
                        int(width),
                        int(height),
                    ],
                    "area": int(area),
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        if (
            image_id % 500 == 0
            or image_id == total_split_images
        ):
            print(
                f"[{split_name}] "
                f"{image_id}/{total_split_images} immagini elaborate"
            )

    coco_dataset = {
        "info": {
            "description": (
                "Football advertising-board detection dataset"
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [COCO_CATEGORY],
    }

    annotations_output_path = (
        split_directory / "_annotations.coco.json"
    )

    with annotations_output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            coco_dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "zero_annotation_images": zero_annotation_images,
        "source_class_counts": dict(source_class_counts),
        "skipped_empty_masks": skipped_empty_masks,
        "skipped_small_boxes": skipped_small_boxes,
        "skipped_invalid_boxes": skipped_invalid_boxes,
        "skipped_unknown_classes": skipped_unknown_classes,
        "size_mismatches": size_mismatches,
        "annotations_file": str(annotations_output_path),
    }


def main() -> None:
    arguments = parse_arguments()

    images_dir = arguments.images_dir.resolve()
    annotations_dir = arguments.annotations_dir.resolve()
    output_dir = arguments.output_dir.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Cartella immagini non trovata: {images_dir}"
        )

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Cartella annotazioni non trovata: "
            f"{annotations_dir}"
        )

    (
        valid_pairs,
        images_without_annotations,
        orphan_annotations,
    ) = find_dataset_pairs(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
    )

    print("\nAnalisi dei file di input")
    print("-------------------------")
    print(f"Coppie valide: {len(valid_pairs)}")
    print(
        "Immagini senza annotazione:",
        len(images_without_annotations),
    )
    print(
        "Annotazioni orfane:",
        len(orphan_annotations),
    )

    splits = split_dataset(
        pairs=valid_pairs,
        train_ratio=arguments.train_ratio,
        valid_ratio=arguments.valid_ratio,
        seed=arguments.seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_summary: dict[str, Any] = {
        "input": {
            "valid_pairs": len(valid_pairs),
            "images_without_annotations": len(
                images_without_annotations
            ),
            "orphan_annotations": len(orphan_annotations),
            "first_orphan_annotations": [
                path.name
                for path in orphan_annotations[:10]
            ],
        },
        "configuration": {
            "train_ratio": arguments.train_ratio,
            "valid_ratio": arguments.valid_ratio,
            "test_ratio": (
                1
                - arguments.train_ratio
                - arguments.valid_ratio
            ),
            "seed": arguments.seed,
            "min_area": arguments.min_area,
            "copy_mode": arguments.copy_mode,
            "merged_category": COCO_CATEGORY,
        },
        "splits": {},
    }

    for split_name, split_pairs in splits.items():
        print(
            f"\nPreparazione split '{split_name}' "
            f"con {len(split_pairs)} immagini"
        )

        split_directory = output_dir / split_name

        reset_split_directory(split_directory)

        split_summary = build_coco_split(
            split_name=split_name,
            pairs=split_pairs,
            split_directory=split_directory,
            min_area=arguments.min_area,
            copy_mode=arguments.copy_mode,
        )

        complete_summary["splits"][
            split_name
        ] = split_summary

    summary_path = (
        output_dir / "preparation_summary.json"
    )

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            complete_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nPreparazione completata")
    print("-----------------------")

    total_output_images = 0
    total_output_annotations = 0
    total_skipped_small_boxes = 0
    total_skipped_invalid_boxes = 0

    for split_name, summary in (
        complete_summary["splits"].items()
    ):
        total_output_images += summary["images"]
        total_output_annotations += summary["annotations"]
        total_skipped_small_boxes += (
            summary["skipped_small_boxes"]
        )
        total_skipped_invalid_boxes += (
            summary["skipped_invalid_boxes"]
        )

        print(
            f"{split_name}: "
            f"{summary['images']} immagini, "
            f"{summary['annotations']} annotazioni"
        )

    print(
        f"\nTotale immagini: {total_output_images}"
    )
    print(
        f"Totale annotazioni COCO: "
        f"{total_output_annotations}"
    )
    print(
        f"Box escluse per area minima: "
        f"{total_skipped_small_boxes}"
    )
    print(
        f"Box escluse perché non valide: "
        f"{total_skipped_invalid_boxes}"
    )
    print(
        f"Riepilogo salvato in: {summary_path}"
    )


if __name__ == "__main__":
    main()