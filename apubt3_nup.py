"""Color APUBT3-NUP image compressor inspired by Zhang, Cai, and Xiong.

This is a practical implementation of the paper's main model:

* ITANRP texture-adaptive non-uniform partitioning
* APUBT3-style block transform coding
* Adaptive scalar quantization for 8x8 and 16x16 blocks
* Color-only reconstruction

The paper-style encoder writes a custom compressed bitstream that stores the
quantized APUBT3-NUP coefficients. Decode that bitstream to an image for
viewing or metric calculation.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def _legendre_u_degree3_basis(n: int) -> np.ndarray:
    """Build a discrete U-system degree-3-like basis and orthogonalize it."""
    xs = (np.arange(n, dtype=np.float64) + 0.5) / n
    rows = [
        np.ones_like(xs),
        math.sqrt(3.0) * (1.0 - 2.0 * xs),
        math.sqrt(5.0) * (6.0 * xs**2 - 6.0 * xs + 1.0),
        math.sqrt(7.0) * (-20.0 * xs**3 + 30.0 * xs**2 - 12.0 * xs + 1.0),
    ]
    while len(rows) < n:
        base = rows[len(rows) % 4]
        freq = len(rows) // 4 + 1
        rows.append(base * np.sign(np.cos(np.pi * freq * xs)))
    mat = np.vstack(rows[:n])
    q, _ = np.linalg.qr(mat.T)
    return q.T


def apubt3_matrix(size: int) -> np.ndarray:
    if size == 8:
        # The printed APUBT3 matrix is kept above for reference, but direct
        # scalar quantization with that non-normalized matrix can amplify
        # reconstruction errors. This stable U-system degree-3 core preserves
        # the paper's transform idea while keeping decoded images close.
        return _legendre_u_degree3_basis(8)
    if size == 16:
        # The paper states that the 16x16 APUBT3 matrix is calculated by the
        # same APDF process, but does not print the matrix. For a stable and
        # reproducible implementation, use the orthogonalized discrete
        # U-system degree-3 basis as the 16x16 transform core.
        return _legendre_u_degree3_basis(16)
    raise ValueError("APUBT3-NUP only uses 8x8 and 16x16 blocks")


def pad_to_multiple(image: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return padded, (h, w)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    clipped = np.clip(ycbcr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(clipped, cv2.COLOR_YCrCb2RGB)


def _fit_plane_rmse(block: np.ndarray) -> float:
    yy, xx = np.mgrid[0 : block.shape[0], 0 : block.shape[1]]
    design = np.column_stack([yy.ravel(), xx.ravel(), np.ones(block.size)])
    coeffs, *_ = np.linalg.lstsq(design, block.ravel(), rcond=None)
    fitted = design @ coeffs
    return float(np.sqrt(np.mean((fitted - block.ravel()) ** 2)))


def itanrp_partition(luma: np.ndarray, epsilon: float) -> list[tuple[int, int, int]]:
    """Return (y, x, size) blocks using the paper's ITANRP texture test.

    Core algorithm comment:
    Every 16x16 luma tile is approximated by z = ax + by + c. If that plane
    fits with low RMSE, the tile is visually smooth and can remain a single
    16x16 block. If the RMSE is high, texture/detail is present, so the tile is
    split into four 8x8 blocks and receives gentler quantization later.
    """
    blocks: list[tuple[int, int, int]] = []
    h, w = luma.shape
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            tile = luma[y : y + 16, x : x + 16]
            if _fit_plane_rmse(tile) < epsilon:
                blocks.append((y, x, 16))
            else:
                blocks.extend((y + dy, x + dx, 8) for dy in (0, 8) for dx in (0, 8))
    return blocks


def zigzag_indices(size: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for s in range(2 * size - 1):
        diagonal = [(i, s - i) for i in range(size) if 0 <= s - i < size]
        if s % 2 == 0:
            diagonal.reverse()
        order.extend(diagonal)
    return order


def rle_encode(values: Iterable[int]) -> list[tuple[int, int]]:
    encoded: list[tuple[int, int]] = []
    last = None
    count = 0
    for value in values:
        value = int(value)
        if value == last:
            count += 1
        else:
            if last is not None:
                encoded.append((count, last))
            last = value
            count = 1
    if last is not None:
        encoded.append((count, last))
    return encoded


def rle_decode(pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    out: list[int] = []
    for count, value in pairs:
        out.extend([int(value)] * int(count))
    return np.array(out, dtype=np.int16)


def _encode_block(block: np.ndarray, matrix: np.ndarray, q: float) -> np.ndarray:
    """Transform and quantize one image block.

    Core algorithm comment:
    APUBT3 decorrelates the block so most visible structure moves toward low
    frequencies. Scalar quantization then drops small high-frequency changes.
    The 8x8 detailed blocks use q1; smoother 16x16 blocks use q2, so smooth
    regions contribute more compression while detail regions keep more bits.
    """
    centered = block.astype(np.float64) - 128.0
    coeffs = matrix @ centered @ matrix.T
    return np.rint(coeffs / q).astype(np.int16)


def _decode_block(values: np.ndarray, matrix: np.ndarray, q: float) -> np.ndarray:
    coeffs = values.astype(np.float64) * q
    inv = np.linalg.pinv(matrix)
    block = inv @ coeffs @ inv.T
    return np.clip(block + 128.0, 0, 255).astype(np.float32)


def adaptive_q2(q1: float, block_count: int, detail_count: int, mu: float = 100.0) -> float:
    if block_count == 0:
        return q1
    ratio = mu * (3 * block_count - 2 * detail_count) / (2 * block_count)
    return max(q1, q1 * ratio / 100.0)


def _rebuild_from_payload(
    payload_blocks: list,
    metadata: dict,
    residual: np.ndarray | None = None,
) -> np.ndarray:
    padded_h, padded_w = metadata["padded_shape"]
    ycbcr = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    matrices = {8: apubt3_matrix(8), 16: apubt3_matrix(16)}
    zigs = {8: zigzag_indices(8), 16: zigzag_indices(16)}

    for channel, y, x, size, encoded in payload_blocks:
        q_base = metadata["q1"] if size == 8 else metadata["q2"]
        q = q_base * (1.0 if channel == 0 else 1.35)
        flat = rle_decode(encoded)
        coeffs = np.zeros((size, size), dtype=np.int16)
        for value, (i, j) in zip(flat, zigs[size]):
            coeffs[i, j] = value
        ycbcr[y : y + size, x : x + size, channel] = _decode_block(coeffs, matrices[size], q)

    h, w = metadata["original_shape"]
    rgb = ycbcr_to_rgb(ycbcr[:h, :w, :]).astype(np.int16)
    if residual is not None:
        rgb = np.clip(rgb + residual.astype(np.int16), 0, 255)
    return rgb.astype(np.uint8)


def _jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=False,
        subsampling=2,
    )
    return buffer.getvalue()


def _write_target_jpeg(rgb: np.ndarray, output_path: Path, max_quality: int, target_size: int) -> tuple[int, int]:
    """Save the highest-quality JPEG that reaches the requested byte target.

    JPEG file size depends on image content and encoder settings, so a fixed
    quality value can accidentally produce weak compression. This loop treats
    the user's quality as an upper bound and searches downward until the output
    reaches the requested target size.
    """
    best_quality = max(1, min(max_quality, 95))
    best_data = _jpeg_bytes(rgb, best_quality)
    if len(best_data) < target_size:
        output_path.write_bytes(best_data)
        return best_quality, len(best_data)

    for jpeg_quality in range(best_quality - 1, 4, -1):
        data = _jpeg_bytes(rgb, jpeg_quality)
        best_quality, best_data = jpeg_quality, data
        if len(data) < target_size:
            break

    output_path.write_bytes(best_data)
    return best_quality, len(best_data)


def _build_payload(
    input_path: Path,
    quality: int = 85,
    epsilon: float | None = None,
    preserve_residual: bool = False,
    target_ratio: float = 0.60,
    q_scale: float = 1.0,
) -> tuple[dict, list, np.ndarray | None]:
    if input_path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("Please provide a JPEG input image with .jpg or .jpeg extension.")
    image = Image.open(input_path).convert("RGB")
    rgb = np.array(image)
    ycbcr, original_shape = pad_to_multiple(rgb_to_ycbcr(rgb), 16)

    # Paper-guided epsilon defaults: lower quality keeps more 16x16 smooth tiles.
    if epsilon is None:
        epsilon = 0.8 if quality < 25 else 0.6 if quality <= 80 else 0.4
    # Lower target ratios need stronger transform quantization before JPEG
    # encoding, otherwise JPEG has to do all the work and artifacts become harsher.
    ratio_pressure = max(0.0, 1.0 - target_ratio)
    q1 = max(1.0, 18.0 - quality * 0.16) * (1.0 + 3.0 * ratio_pressure) * q_scale

    blocks = itanrp_partition(ycbcr[:, :, 0], epsilon)
    lsize = (ycbcr.shape[0] // 16) * (ycbcr.shape[1] // 16)
    detail_tiles = sum(1 for _, _, size in blocks if size == 8) // 4
    q2 = adaptive_q2(q1, lsize, detail_tiles)
    matrices = {8: apubt3_matrix(8), 16: apubt3_matrix(16)}
    zigs = {8: zigzag_indices(8), 16: zigzag_indices(16)}

    payload_blocks = []
    for channel in range(3):
        chroma_scale = 1.0 if channel == 0 else 1.35
        for y, x, size in blocks:
            q = (q1 if size == 8 else q2) * chroma_scale
            coeffs = _encode_block(ycbcr[y : y + size, x : x + size, channel], matrices[size], q)
            flat = [coeffs[i, j] for i, j in zigs[size]]
            payload_blocks.append((channel, y, x, size, rle_encode(flat)))

    metadata = {
        "original_shape": original_shape,
        "padded_shape": ycbcr.shape[:2],
        "quality": quality,
        "epsilon": epsilon,
        "q1": q1,
        "q2": q2,
        "blocks": len(blocks),
        "base_tiles": lsize,
        "detail_tiles": detail_tiles,
        "detail_blocks": detail_tiles * 4,
        "preserve_residual": preserve_residual,
        "target_ratio": target_ratio,
        "q_scale": q_scale,
    }

    residual = None
    if preserve_residual:
        # Core algorithm comment:
        # The transform-coded image is the paper implementation. This residual
        # layer stores the small remaining RGB error after reconstruction, so
        # the final color image stays visually similar to the input instead of
        # showing strong color drift or block distortion.
        base_rgb = _rebuild_from_payload(payload_blocks, metadata)
        residual = rgb.astype(np.int16) - base_rgb.astype(np.int16)

    return metadata, payload_blocks, residual


def encode_apubt3(
    input_path: Path,
    output_path: Path,
    quality: int = 85,
    epsilon: float | None = None,
    preserve_residual: bool = False,
    target_ratio: float = 0.60,
) -> dict:
    target_bytes = max(1024, int(input_path.stat().st_size * target_ratio))
    scale = 1.0
    metadata = {}
    for _ in range(8):
        metadata, payload_blocks, residual = _build_payload(
            input_path,
            quality,
            epsilon,
            preserve_residual,
            target_ratio,
            q_scale=scale,
        )
        arrays = _payload_to_arrays(metadata, payload_blocks, residual)
        with output_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        if output_path.stat().st_size <= target_bytes:
            break
        scale *= 1.75

    h, w = metadata["original_shape"]
    metadata["original_bytes"] = input_path.stat().st_size
    metadata["target_bytes"] = target_bytes
    metadata["compressed_bytes"] = output_path.stat().st_size
    metadata["bpp"] = output_path.stat().st_size * 8 / (h * w)
    metadata["compression_ratio"] = input_path.stat().st_size / output_path.stat().st_size
    metadata["target_reached"] = output_path.stat().st_size <= target_bytes
    return metadata


def _payload_to_arrays(metadata: dict, payload_blocks: list, residual: np.ndarray | None) -> dict:
    block_count = metadata["blocks"]
    block_y = np.array([item[1] for item in payload_blocks[:block_count]], dtype=np.uint16)
    block_x = np.array([item[2] for item in payload_blocks[:block_count]], dtype=np.uint16)
    block_size = np.array([item[3] for item in payload_blocks[:block_count]], dtype=np.uint8)
    lengths = np.zeros((3, block_count), dtype=np.uint16)
    packed_values: list[int] = []

    packed_values = []
    for channel in range(3):
        channel_blocks = payload_blocks[channel * block_count : (channel + 1) * block_count]
        for block_index, item in enumerate(channel_blocks):
            flat = rle_decode(item[4])
            nonzero = np.flatnonzero(flat)
            keep = int(nonzero[-1] + 1) if nonzero.size else 1
            trimmed = flat[:keep].astype(np.int16)
            lengths[channel, block_index] = keep
            packed_values.extend(int(value) for value in trimmed)

    arrays = {
        "metadata": json.dumps(metadata),
        "block_y": block_y,
        "block_x": block_x,
        "block_size": block_size,
        "lengths": lengths,
        "values": np.array(packed_values, dtype=np.int16),
    }
    if residual is not None:
        arrays["residual"] = residual.astype(np.int16)
    return arrays


def decode_apubt3_array(input_path: Path) -> tuple[np.ndarray, dict]:
    archive = np.load(input_path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    residual = archive["residual"] if "residual" in archive.files else None

    block_y = archive["block_y"]
    block_x = archive["block_x"]
    block_size = archive["block_size"]
    lengths = archive["lengths"]
    values = archive["values"]
    payload_blocks = []
    cursor = 0
    for channel in range(3):
        for block_index, size in enumerate(block_size):
            size_int = int(size)
            coeff_count = size_int * size_int
            keep = int(lengths[channel, block_index])
            flat = np.zeros(coeff_count, dtype=np.int16)
            flat[:keep] = values[cursor : cursor + keep]
            cursor += keep
            payload_blocks.append(
                (
                    channel,
                    int(block_y[block_index]),
                    int(block_x[block_index]),
                    size_int,
                    rle_encode(flat),
                )
            )

    rgb = _rebuild_from_payload(payload_blocks, metadata, residual=residual)
    h, w = metadata["original_shape"]
    metadata["compressed_bytes"] = input_path.stat().st_size
    metadata["bpp"] = input_path.stat().st_size * 8 / (h * w)
    return rgb, metadata


def decode_apubt3(input_path: Path, output_path: Path) -> dict:
    rgb, metadata = decode_apubt3_array(input_path)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    return metadata



def export_jpeg(
    input_path: Path,
    output_path: Path,
    quality: int = 85,
    epsilon: float | None = None,
    preserve_residual: bool = False,
    target_ratio: float = 0.60,
) -> dict:
    if output_path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("JPEG export output must use .jpg or .jpeg extension.")
    original_bytes = input_path.stat().st_size
    metadata, payload_blocks, residual = _build_payload(input_path, quality, epsilon, preserve_residual, target_ratio)
    compressed_rgb = _rebuild_from_payload(payload_blocks, metadata, residual=residual)
    target_bytes = max(1024, int(original_bytes * target_ratio))
    jpeg_quality, compressed_bytes = _write_target_jpeg(compressed_rgb, output_path, quality, target_bytes)
    metadata["original_bytes"] = original_bytes
    metadata["target_bytes"] = target_bytes
    metadata["compressed_bytes"] = compressed_bytes
    metadata["jpeg_save_quality"] = jpeg_quality
    metadata["compression_ratio"] = original_bytes / compressed_bytes
    metadata["is_smaller_than_original"] = compressed_bytes < original_bytes
    metadata["target_reached"] = compressed_bytes <= target_bytes
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="APUBT3-NUP color image compressor")
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("encode", help="encode JPEG to paper-style APUBT3-NUP bitstream")
    e.add_argument("input", type=Path)
    e.add_argument("output", type=Path)
    e.add_argument("--quality", type=int, default=85, choices=range(1, 101), metavar="1-100")
    e.add_argument("--epsilon", type=float, default=None)
    e.add_argument("--preserve-residual", action="store_true", help="store a correction layer in the bitstream")
    e.add_argument(
        "--target-ratio",
        type=float,
        default=0.60,
        help="target compressed/original file-size ratio, for example 0.50 means about half size",
    )

    d = sub.add_parser("decode", help="decode APUBT3-NUP bitstream to a viewable image")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)

    x = sub.add_parser("export-jpeg", help="run APUBT3-NUP then save the result as JPEG for viewing")
    x.add_argument("input", type=Path)
    x.add_argument("output", type=Path)
    x.add_argument("--quality", type=int, default=85, choices=range(1, 101), metavar="1-100")
    x.add_argument("--epsilon", type=float, default=None)
    x.add_argument("--preserve-residual", action="store_true", help="apply a correction layer before saving the JPEG")
    x.add_argument("--target-ratio", type=float, default=0.60)

    args = parser.parse_args()
    if hasattr(args, "target_ratio") and not 0.05 <= args.target_ratio < 1.0:
        raise ValueError("--target-ratio must be between 0.05 and 0.99")
    if args.command == "encode":
        meta = encode_apubt3(
            args.input,
            args.output,
            args.quality,
            args.epsilon,
            preserve_residual=args.preserve_residual,
            target_ratio=args.target_ratio,
        )
    elif args.command == "decode":
        meta = decode_apubt3(args.input, args.output)
    else:
        meta = export_jpeg(
            args.input,
            args.output,
            args.quality,
            args.epsilon,
            preserve_residual=args.preserve_residual,
            target_ratio=args.target_ratio,
        )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()