"""Portable pixel preprocessing; identical v1 pixels without PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageOps



ROOT = Path(__file__).resolve().parent
Polarity = Literal["auto", "dark_on_light", "light_on_dark"]
ImageSource = str | Path | Image.Image | np.ndarray


@dataclass
class PreprocessResult:
    """The final model input plus intermediate images for debugging."""

    original_gray: np.ndarray
    contrast_image: np.ndarray
    cropped_image: np.ndarray
    normalized_uint8: np.ndarray
    metadata: dict


def _to_grayscale_uint8(source: ImageSource) -> np.ndarray:
    """Read supported input types and return an owned HxW uint8 grayscale array."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).copy()
    elif isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source).copy()
    elif isinstance(source, np.ndarray):
        array = np.asarray(source)
        if array.size == 0:
            raise ValueError("Input NumPy image is empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Input NumPy image must contain numeric pixels")
        if not np.isfinite(array).all():
            raise ValueError("Input NumPy image contains NaN or infinity")
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0 and array.min() >= 0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
        if array.ndim == 2:
            image = Image.fromarray(array, mode="L")
        elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
            if array.shape[2] == 1:
                image = Image.fromarray(array[:, :, 0], mode="L")
            else:
                image = Image.fromarray(array, mode="RGB" if array.shape[2] == 3 else "RGBA")
        else:
            raise ValueError("NumPy image shape must be HxW, HxWx1, HxWx3, or HxWx4")
    else:
        raise TypeError("source must be a path, PIL.Image, or NumPy array")

    # Transparent pixels are treated as white paper instead of becoming black.
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    gray = image.convert("L")
    if gray.width < 2 or gray.height < 2:
        raise ValueError("Image must be at least 2x2 pixels")
    return np.asarray(gray, dtype=np.uint8).copy()


def _border_pixels(image: np.ndarray) -> np.ndarray:
    border = max(1, int(round(min(image.shape) * 0.05)))
    return np.concatenate(
        [
            image[:border, :].ravel(),
            image[-border:, :].ravel(),
            image[:, :border].ravel(),
            image[:, -border:].ravel(),
        ]
    )


def _shift_without_wrap(image: np.ndarray, row_shift: int, column_shift: int) -> np.ndarray:
    """Translate an image while filling new pixels with zero; never wrap at edges."""

    output = np.zeros_like(image)
    height, width = image.shape
    source_r0 = max(0, -row_shift)
    source_r1 = min(height, height - row_shift)
    source_c0 = max(0, -column_shift)
    source_c1 = min(width, width - column_shift)
    target_r0 = source_r0 + row_shift
    target_r1 = source_r1 + row_shift
    target_c0 = source_c0 + column_shift
    target_c1 = source_c1 + column_shift
    if source_r1 > source_r0 and source_c1 > source_c0:
        output[target_r0:target_r1, target_c0:target_c1] = image[
            source_r0:source_r1, source_c0:source_c1
        ]
    return output


def preprocess_digit_image(
    source: ImageSource,
    *,
    polarity: Polarity = "auto",
    output_size: int = 28,
    content_size: int = 19,
    minimum_contrast: float = 20.0,
) -> PreprocessResult:
    """Normalize one isolated digit to a centered white-on-black 28x28 image.

    The algorithm estimates the background from the outer border, makes the digit
    bright, finds its bounding box, preserves aspect ratio while fitting it in a
    19x19 area, then shifts its center of mass to the middle of a 28x28 canvas.
    """

    if polarity not in ("auto", "dark_on_light", "light_on_dark"):
        raise ValueError("polarity must be auto, dark_on_light, or light_on_dark")
    if not (2 <= content_size <= output_size):
        raise ValueError("Require 2 <= content_size <= output_size")

    original = _to_grayscale_uint8(source)
    # A small median filter suppresses isolated camera noise without altering MNIST-size inputs.
    if min(original.shape) >= 64:
        working = np.asarray(
            Image.fromarray(original, mode="L").filter(ImageFilter.MedianFilter(size=3)),
            dtype=np.uint8,
        )
    else:
        working = original

    background = float(np.median(_border_pixels(working)))
    resolved_polarity = polarity
    if polarity == "auto":
        resolved_polarity = "dark_on_light" if background >= 127.5 else "light_on_dark"

    working_float = working.astype(np.float32)
    if resolved_polarity == "dark_on_light":
        strength = np.clip(background - working_float, 0.0, 255.0)
    else:
        strength = np.clip(working_float - background, 0.0, 255.0)

    robust_peak = float(np.percentile(strength, 99.5))
    if robust_peak < minimum_contrast:
        raise ValueError(
            f"No clear digit found: contrast {robust_peak:.1f} is below {minimum_contrast:.1f}"
        )
    strength = np.clip(strength * (255.0 / robust_peak), 0.0, 255.0)
    contrast_image = np.rint(strength).astype(np.uint8)

    # A native MNIST/PYNQ contract image is already in the exact model format.
    # Keeping it byte-for-byte avoids needless interpolation and accuracy loss.
    canonical_input = (
        original.shape == (output_size, output_size)
        and resolved_polarity == "light_on_dark"
        and background <= 8.0
    )
    if canonical_input:
        contrast_image = original.copy()
        strength = contrast_image.astype(np.float32)

    # The mask only locates the digit; gray antialiased pixels are retained in the crop.
    threshold = 255.0 * 0.05
    mask = strength >= threshold
    rows, columns = np.nonzero(mask)
    if rows.size < 4:
        raise ValueError("No clear digit found: fewer than four foreground pixels")

    row0, row1 = int(rows.min()), int(rows.max()) + 1
    column0, column1 = int(columns.min()), int(columns.max()) + 1
    box_height, box_width = row1 - row0, column1 - column0
    cropped = contrast_image[row0:row1, column0:column1]

    if canonical_input:
        resized_height, resized_width = cropped.shape
        row_shift, column_shift = 0, 0
        normalized = original.copy()
        normalization_action = "kept exact canonical 28x28 input"
    else:
        scale = content_size / max(cropped.shape)
        resized_height = max(1, int(round(cropped.shape[0] * scale)))
        resized_width = max(1, int(round(cropped.shape[1] * scale)))
        resized = np.asarray(
            Image.fromarray(cropped, mode="L").resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
        canvas = np.zeros((output_size, output_size), dtype=np.uint8)
        paste_row = (output_size - resized_height) // 2
        paste_column = (output_size - resized_width) // 2
        canvas[
            paste_row : paste_row + resized_height,
            paste_column : paste_column + resized_width,
        ] = resized

        mass = canvas.astype(np.float64)
        total_mass = float(mass.sum())
        if total_mass <= 0:
            raise ValueError("Digit disappeared during resizing")
        row_center = float((mass.sum(axis=1) * np.arange(output_size)).sum() / total_mass)
        column_center = float((mass.sum(axis=0) * np.arange(output_size)).sum() / total_mass)
        # MNIST's conventional centering targets coordinate size/2 (14 for 28x28).
        target_center = output_size / 2
        row_shift = int(round(target_center - row_center))
        column_shift = int(round(target_center - column_center))
        normalized = _shift_without_wrap(canvas, row_shift, column_shift)
        normalization_action = "cropped, resized with bicubic interpolation, and centered"

    metadata = {
        "source_shape_hw": [int(original.shape[0]), int(original.shape[1])],
        "source_dtype": str(original.dtype),
        "background_gray": background,
        "resolved_polarity": resolved_polarity,
        "robust_contrast": robust_peak,
        "foreground_threshold_normalized": threshold,
        "crop_box_rc_exclusive": [row0, column0, row1, column1],
        "cropped_shape_hw": [int(cropped.shape[0]), int(cropped.shape[1])],
        "resized_shape_hw": [resized_height, resized_width],
        "center_shift_rc": [row_shift, column_shift],
        "canonical_input_bypassed_resampling": canonical_input,
        "normalization_action": normalization_action,
        "output_shape_hw": [output_size, output_size],
        "output_layout_for_handoff": "row-major uint8, white digit on black background",
    }
    return PreprocessResult(
        original_gray=original,
        contrast_image=contrast_image,
        cropped_image=cropped.copy(),
        normalized_uint8=normalized,
        metadata=metadata,
    )
