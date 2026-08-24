from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT / "data" / "processed" / "rf_detr"
)

SPLIT_NAMES = ("train", "valid", "test")

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def load_coco_annotations(
    annotations_path: Path,
) -> dict[str, Any]:
    with annotations_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_split(
    split_name: str,
    split_directory: Path,
) -> tuple[dict[str, int], set[str]]:
    annotations_path = (
        split_directory / "_annotations.coco.json"
    )

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"File COCO non trovato: {annotations_path}"
        )

    coco_data = load_coco_annotations(annotations_path)

    errors: list[str] = []

    for required_key in (
        "images",
        "annotations",
        "categories",
    ):
        if required_key not in coco_data:
            errors.append(
                f"Chiave COCO mancante: {required_key}"
            )

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    if len(categories) != 1:
        errors.append(
            "Il dataset dovrebbe contenere una sola categoria."
        )
    else:
        category = categories[0]

        if category.get("id") != 1:
            errors.append(
                "L'id della categoria dovrebbe essere 1."
            )

        if category.get("name") != "advertising_board":
            errors.append(
                "La categoria dovrebbe chiamarsi "
                "'advertising_board'."
            )

    image_ids = [image["id"] for image in images]
    annotation_ids = [
        annotation["id"]
        for annotation in annotations
    ]

    if len(image_ids) != len(set(image_ids)):
        errors.append(
            "Sono presenti image_id duplicati."
        )

    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append(
            "Sono presenti annotation_id duplicati."
        )

    image_by_id = {
        image["id"]: image
        for image in images
    }

    coco_file_names = {
        image["file_name"]
        for image in images
    }

    physical_file_names = {
        path.name
        for path in split_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    }

    missing_files = (
        coco_file_names - physical_file_names
    )

    extra_files = (
        physical_file_names - coco_file_names
    )

    if missing_files:
        errors.append(
            f"{len(missing_files)} immagini indicate nel COCO "
            "non sono presenti nella cartella."
        )

    if extra_files:
        errors.append(
            f"{len(extra_files)} immagini presenti nella cartella "
            "non sono indicate nel COCO."
        )

    annotations_per_image: dict[int, int] = {
        image_id: 0
        for image_id in image_ids
    }

    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")

        if image_id not in image_by_id:
            errors.append(
                f"Annotazione {annotation.get('id')} associata "
                f"a image_id inesistente: {image_id}"
            )
            continue

        annotations_per_image[image_id] += 1

        if category_id != 1:
            errors.append(
                f"Categoria non valida nell'annotazione "
                f"{annotation.get('id')}: {category_id}"
            )

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            errors.append(
                f"Bounding box non valida nell'annotazione "
                f"{annotation.get('id')}."
            )
            continue

        x, y, width, height = bbox

        image_data = image_by_id[image_id]
        image_width = image_data["width"]
        image_height = image_data["height"]

        if width <= 0 or height <= 0:
            errors.append(
                f"Bounding box degenere nell'annotazione "
                f"{annotation.get('id')}."
            )

        if x < 0 or y < 0:
            errors.append(
                f"Coordinate negative nell'annotazione "
                f"{annotation.get('id')}."
            )

        if x + width > image_width:
            errors.append(
                f"Box oltre la larghezza dell'immagine "
                f"nell'annotazione {annotation.get('id')}."
            )

        if y + height > image_height:
            errors.append(
                f"Box oltre l'altezza dell'immagine "
                f"nell'annotazione {annotation.get('id')}."
            )

        expected_area = width * height
        stored_area = annotation.get("area")

        if stored_area != expected_area:
            errors.append(
                f"Area non coerente nell'annotazione "
                f"{annotation.get('id')}."
            )

    zero_annotation_images = sum(
        count == 0
        for count in annotations_per_image.values()
    )

    if errors:
        print(f"\n[{split_name}] ERRORI TROVATI")

        for error in errors[:20]:
            print("-", error)

        if len(errors) > 20:
            print(
                f"... altri {len(errors) - 20} errori"
            )

        raise ValueError(
            f"Validazione fallita per lo split {split_name}."
        )

    print(f"\n[{split_name}] validazione completata")
    print(f"Immagini: {len(images)}")
    print(f"Annotazioni: {len(annotations)}")
    print(f"immagini senza annotazioni: {zero_annotation_images}")
    print(f"Categorie: {categories}")

    return (
        {
            "images": len(images),
            "annotations": len(annotations),
            "zero_annotation_images": zero_annotation_images,
        },
        coco_file_names,
    )


def main() -> None:
    all_split_files: dict[str, set[str]] = {}
    summaries: dict[str, dict[str, int]] = {}

    for split_name in SPLIT_NAMES:
        split_directory = DATASET_DIR / split_name

        if not split_directory.exists():
            raise FileNotFoundError(
                f"Split non trovato: {split_directory}"
            )

        summary, file_names = validate_split(
            split_name=split_name,
            split_directory=split_directory,
        )

        summaries[split_name] = summary
        all_split_files[split_name] = file_names

    train_valid_overlap = (
        all_split_files["train"]
        & all_split_files["valid"]
    )

    train_test_overlap = (
        all_split_files["train"]
        & all_split_files["test"]
    )

    valid_test_overlap = (
        all_split_files["valid"]
        & all_split_files["test"]
    )

    if (
        train_valid_overlap
        or train_test_overlap
        or valid_test_overlap
    ):
        raise ValueError(
            "Sono presenti immagini duplicate tra gli split."
        )

    total_images = sum(
        summary["images"]
        for summary in summaries.values()
    )

    total_annotations = sum(
        summary["annotations"]
        for summary in summaries.values()
    )

    total_zero_annotation_images = sum(
        summary["zero_annotation_images"]
        for summary in summaries.values()
    )

    print("\nValidazione globale completata")
    print("------------------------------")
    print(f"Totale immagini: {total_images}")
    print(
        f"Totale annotazioni: {total_annotations}"
    )
    print(
        f"Totale immagini senza annotazioni: "
        f"{total_zero_annotation_images}"
    )
    print("Duplicati tra gli split: 0")


if __name__ == "__main__":
    main()