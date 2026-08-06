from __future__ import annotations

import base64
import io
import zlib

import numpy as np
from PIL import Image


def decode_supervisely_bitmap(data: str) -> np.ndarray:
    """
    Decodifica una bitmap Supervisely compressa in una maschera booleana.

    Parameters
    ----------
    data:
        Stringa Base64 contenente una PNG compressa con zlib.

    Returns
    -------
    np.ndarray
        Maschera booleana bidimensionale.
    """
    compressed_data = base64.b64decode(data)
    png_bytes = zlib.decompress(compressed_data)

    mask_image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    mask_array = np.asarray(mask_image)

    return mask_array[:, :, 3] > 0


def get_object_bbox(
    obj: dict,
) -> tuple[int, int, int, int] | None:
    """
    Converte una bitmap Supervisely nella relativa bounding box assoluta.

    La bounding box restituita usa il formato:

        (x_min, y_min, x_max, y_max)

    con coordinate massime inclusive.

    Parameters
    ----------
    obj:
        Oggetto Supervisely contenente i campi bitmap.data e bitmap.origin.

    Returns
    -------
    tuple[int, int, int, int] | None
        Bounding box assoluta oppure None se la maschera è vuota.
    """
    bitmap = obj.get("bitmap")

    if bitmap is None:
        return None

    local_mask = decode_supervisely_bitmap(bitmap["data"])

    ys, xs = np.where(local_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    origin_x, origin_y = bitmap["origin"]

    x_min = int(origin_x + xs.min())
    y_min = int(origin_y + ys.min())
    x_max = int(origin_x + xs.max())
    y_max = int(origin_y + ys.max())

    return x_min, y_min, x_max, y_max


def bbox_xyxy_to_coco(
    bbox: tuple[int, int, int, int],
) -> list[int]:
    """
    Converte una bounding box da XYXY al formato COCO XYWH.

    COCO utilizza:

        [x_min, y_min, width, height]
    """
    x_min, y_min, x_max, y_max = bbox

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [x_min, y_min, width, height]


def is_valid_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_area: int = 100,
) -> bool:
    """
    Verifica che una bounding box sia interna all'immagine e non degenere.
    """
    x_min, y_min, x_max, y_max = bbox

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = width * height

    return (
        x_min >= 0
        and y_min >= 0
        and x_max < image_width
        and y_max < image_height
        and width > 0
        and height > 0
        and area >= min_area
    )