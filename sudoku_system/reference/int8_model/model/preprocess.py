"""单格灰度、裁剪、缩放和居中；两种入口共用同一套像素处理规则。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps


Polarity = Literal["auto", "dark_on_light", "light_on_dark"]
ImageSource = Union[str, Path, Image.Image, np.ndarray]
# 兼容板上 Python 3.8 和旧版 Pillow；新旧名称均为同一种双三次插值。
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC


class NoForegroundError(ValueError):
    """没有清晰笔画，用于区分空白候选和文件读取错误。"""


@dataclass
class PreprocessResult:
    """处理前后图片、最终UINT8输入及处理说明。"""

    original_gray: np.ndarray
    contrast_image: np.ndarray
    cropped_image: np.ndarray
    normalized_uint8: np.ndarray
    metadata: dict


# 读图并统一灰度：透明背景按白色处理，浮点0～1数组还原到0～255
def _to_grayscale_uint8(source: ImageSource) -> np.ndarray:
    if isinstance(source, (str, Path)):
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).copy()
    elif isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source).copy()
    else:
        array = np.asarray(source)
        if not np.isfinite(array).all():
            raise ValueError("Input NumPy image contains NaN or infinity")
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0 and array.min() >= 0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        image = Image.fromarray(array)
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
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
    """平移图片并在空出位置补零，禁止越界像素绕到另一侧。"""

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


# 普通数字入口：只转换一次灰度；没有笔画时仍抛出NoForegroundError
def preprocess_digit_image(
    source: ImageSource, *, polarity: Polarity = "auto",
    output_size: int = 28, content_size: int = 19, minimum_contrast: float = 20.0,
) -> PreprocessResult:
    return _normalize_gray(_to_grayscale_uint8(source), polarity, output_size, content_size, minimum_contrast)


# 两个入口共用的灰度处理，不重复读图或转换数组
def _normalize_gray(
    original: np.ndarray, polarity: Polarity,
    output_size: int = 28, content_size: int = 19, minimum_contrast: float = 20.0,
    foreground_fraction: float = 0.05,
) -> PreprocessResult:
    # 大图先用 3×3 中值滤波抑制孤立噪声；28×28 输入不经过此滤波。
    if min(original.shape) >= 64:
        working = np.asarray(
            Image.fromarray(original, mode="L").filter(ImageFilter.MedianFilter(size=3)),
            dtype=np.uint8,
        )
    else:
        working = original

    # 使用边缘中位数估计背景，判断黑字白底或白字黑底。
    background = float(np.median(_border_pixels(working)))
    resolved_polarity = polarity
    if polarity == "auto":
        resolved_polarity = "dark_on_light" if background >= 127.5 else "light_on_dark"

    working_float = working.astype(np.float32)
    if resolved_polarity == "dark_on_light":
        strength = np.clip(background - working_float, 0.0, 255.0)
    else:
        strength = np.clip(working_float - background, 0.0, 255.0)

    # 用 99.5 分位估计笔画强度，减少极少数噪声像素的影响。
    robust_peak = float(np.percentile(strength, 99.5))
    if robust_peak < minimum_contrast:
        raise NoForegroundError(
            f"No clear digit found: contrast {robust_peak:.1f} is below {minimum_contrast:.1f}"
        )
    strength = np.clip(strength * (255.0 / robust_peak), 0.0, 255.0)
    contrast_image = np.rint(strength).astype(np.uint8)

    # 已满足黑边条件的 28×28 白字黑底图片保持原字节，避免重复插值。
    canonical_input = (
        original.shape == (output_size, output_size)
        and resolved_polarity == "light_on_dark"
        and background <= 8.0
    )
    if canonical_input:
        contrast_image = original.copy()
        strength = contrast_image.astype(np.float32)

    # 掩码只确定裁剪边界；提高阈值可减少纸纹干扰，裁剪内容仍保留灰度。
    threshold = 255.0 * foreground_fraction
    mask = strength >= threshold
    rows, columns = np.nonzero(mask)
    if rows.size < 4:
        raise NoForegroundError("No clear digit found: fewer than four foreground pixels")

    row0, row1 = int(rows.min()), int(rows.max()) + 1
    column0, column1 = int(columns.min()), int(columns.max()) + 1
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
                (resized_width, resized_height), BICUBIC
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
        # 将笔画灰度重心移到画布中心；28×28 的目标坐标为 14。
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


def preprocess_sudoku_cell(source: ImageSource, *, polarity: Polarity = "auto") -> PreprocessResult:
    """数独单格入口：标准28×28输入保持原像素，其余处理后输出；淡字可能成为空白候选。"""
    original = _to_grayscale_uint8(source)
    background = float(np.median(_border_pixels(original)))
    resolved = ("dark_on_light" if background >= 127.5 else "light_on_dark") if polarity == "auto" else polarity
    if original.shape == (28,28) and resolved == "light_on_dark" and background <= 8:
        return PreprocessResult(original,original.copy(),original.copy(),original.copy(),
                                {"profile":"sudoku_cell_v2","normalization_action":"canonical bytes preserved",
                                 "resolved_polarity":resolved})
    try:
        # 数独照片用20%边界阈值，避免背景扩大裁剪框；旧数字入口仍用5%。
        result = _normalize_gray(original, polarity, foreground_fraction=0.20)
        result.metadata['profile'] = 'sudoku_cell_v2'
        return result
    except NoForegroundError as error:
        blank = np.zeros((28,28), dtype=np.uint8)
        return PreprocessResult(original,blank.copy(),blank.copy(),blank,
                                {"profile":"sudoku_cell_v2","resolved_polarity":resolved,
                                 "warning":"未检测到足够对比度的笔画；按空白候选输入。淡字可能被漏检，需检查原图。",
                                 "reason":str(error)})


# 单独导出FPGA输入，不运行CNN；PNG用于查看，HEX用于读取784个UINT8像素。
def export_preprocessed_images(source, output_directory=None, *, polarity: Polarity = "auto"):
    source = Path(source).resolve()
    output = Path(output_directory).resolve() if output_directory else Path(__file__).resolve().parent / '预处理图像'
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    root = source.parent if source.is_file() else source
    files = [source] if source.is_file() else sorted(
        p for p in source.rglob('*')
        if p.is_file() and p.suffix.lower() in extensions and output not in p.resolve().parents
    )
    if not files:
        raise FileNotFoundError(f'没有找到单格图片：{source}')
    relative_paths = [p.relative_to(root).with_suffix('.png') for p in files]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError('同目录存在同名不同扩展名图片，请先区分名称，避免导出覆盖。')
    exported = []
    for path, relative in zip(files, relative_paths):
        pixels = preprocess_sudoku_cell(path, polarity=polarity).normalized_uint8
        png_path = output / 'PNG' / relative
        hex_path = output / 'HEX' / relative.with_suffix('.hex')
        png_path.parent.mkdir(parents=True, exist_ok=True)
        hex_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(png_path)
        with hex_path.open('w', encoding='ascii', newline='\n') as stream:
            stream.write(''.join(f'{int(value):02X}\n' for value in pixels.ravel(order='C')))
        exported.append((path, png_path, hex_path))
    return exported


# 在本目录执行：python -B preprocess.py --input "单格图片或数据集目录"
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='导出28×28灰度PNG和FPGA逐像素HEX，不运行CNN')
    parser.add_argument('--input', type=Path, required=True, help='裁好的单格图片或其目录，支持子目录')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / '预处理图像')
    parser.add_argument('--polarity', choices=['auto', 'dark_on_light', 'light_on_dark'], default='auto')
    args = parser.parse_args()
    items = export_preprocessed_images(args.input, args.output, polarity=args.polarity)
    print(f'已导出 {len(items)} 张；PNG：{args.output / "PNG"}；HEX：{args.output / "HEX"}')
