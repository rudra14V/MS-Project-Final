"""Generate paper-style bitrate graphs for APUBT3-NUP.

The script evaluates each JPEG test image at several target file-size ratios,
then saves:

* PSNR vs BPP graph
* SSIM vs BPP graph
* CSV table with the measured values

It compares the implemented APUBT3-NUP compressor with a baseline JPEG encoder.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from apubt3_nup import _write_target_jpeg, decode_apubt3_array, encode_apubt3


DEFAULT_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float64)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute global grayscale SSIM without extra dependencies."""
    x = rgb_to_luma(original)
    y = rgb_to_luma(reconstructed)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ux = float(np.mean(x))
    uy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - ux) * (y - uy)))
    numerator = (2 * ux * uy + c1) * (2 * cov + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def bpp(path: Path, shape: tuple[int, int, int]) -> float:
    h, w = shape[:2]
    return float(path.stat().st_size * 8 / (h * w))


def discover_images(input_dir: Path) -> list[Path]:
    images = []
    for path in sorted(input_dir.glob("*.jp*g")):
        if path.name.lower().startswith("compressed"):
            continue
        images.append(path)
    return images


def save_baseline_jpeg(original_rgb: np.ndarray, input_path: Path, output_path: Path, quality: int, ratio: float) -> dict:
    target_bytes = max(1024, int(input_path.stat().st_size * ratio))
    jpeg_quality, compressed_bytes = _write_target_jpeg(original_rgb, output_path, quality, target_bytes)
    return {
        "original_bytes": input_path.stat().st_size,
        "target_bytes": target_bytes,
        "compressed_bytes": compressed_bytes,
        "jpeg_save_quality": jpeg_quality,
        "compression_ratio": compressed_bytes / input_path.stat().st_size,
    }


def evaluate_image(image_path: Path, ratios: list[float], quality: int, temp_dir: Path) -> list[dict]:
    original = load_rgb(image_path)
    rows = []
    for ratio in ratios:
        apubt3_out = temp_dir / f"{image_path.stem}_apubt3_{ratio:.2f}.apubt3"
        jpeg_out = temp_dir / f"{image_path.stem}_jpeg_{ratio:.2f}.jpg"

        apubt3_meta = encode_apubt3(image_path, apubt3_out, quality=quality, target_ratio=ratio)
        apubt3_reconstructed, _ = decode_apubt3_array(apubt3_out)
        jpeg_meta = save_baseline_jpeg(original, image_path, jpeg_out, quality, ratio)
        jpeg_reconstructed = load_rgb(jpeg_out)

        for method, output_path, reconstructed, meta in [
            ("APUBT3-NUP", apubt3_out, apubt3_reconstructed, apubt3_meta),
            ("JPEG", jpeg_out, jpeg_reconstructed, jpeg_meta),
        ]:
            rows.append(
                {
                    "image": image_path.stem,
                    "method": method,
                    "target_ratio": ratio,
                    "bpp": bpp(output_path, original.shape),
                    "psnr": psnr(original, reconstructed),
                    "ssim": ssim(original, reconstructed),
                    "original_bytes": meta["original_bytes"],
                    "compressed_bytes": meta["compressed_bytes"],
                    "jpeg_save_quality": meta.get("jpeg_save_quality", ""),
                    "target_reached": meta.get("target_reached", ""),
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "image",
        "method",
        "target_ratio",
        "bpp",
        "psnr",
        "ssim",
        "original_bytes",
        "compressed_bytes",
        "jpeg_save_quality",
        "target_reached",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_rows(rows: list[dict]) -> dict[tuple[str, float], dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["target_ratio"]), []).append(row)

    averaged = {}
    for key, values in grouped.items():
        averaged[key] = {
            "bpp": float(np.mean([v["bpp"] for v in values])),
            "psnr": float(np.mean([v["psnr"] for v in values])),
            "ssim": float(np.mean([v["ssim"] for v in values])),
        }
    return averaged


def plot_metric(rows: list[dict], output_path: Path, metric: str, title: str, ylabel: str) -> None:
    averaged = average_rows(rows)
    methods = sorted({row["method"] for row in rows})
    plt.figure(figsize=(8, 5), dpi=150)
    for method in methods:
        points = sorted(
            ((value["bpp"], value[metric]) for key, value in averaged.items() if key[0] == method),
            key=lambda item: item[0],
        )
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        marker = "o" if method == "APUBT3-NUP" else "s"
        plt.plot(xs, ys, marker=marker, linewidth=2, label=method)

    plt.title(title)
    plt.xlabel("Bits per pixel (BPP)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PSNR/SSIM bitrate graphs")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="folder containing input .jpg files")
    parser.add_argument("--output-dir", type=Path, default=Path("bitrate_graphs"))
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=DEFAULT_RATIOS,
        help="target compressed/original size ratios",
    )
    args = parser.parse_args()

    images = discover_images(args.input_dir)
    if not images:
        raise ValueError(f"No JPEG images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="apubt3_graphs_") as temp_name:
        temp_dir = Path(temp_name)
        rows = []
        for image_path in images:
            rows.extend(evaluate_image(image_path, args.ratios, args.quality, temp_dir))

    write_csv(rows, args.output_dir / "bitrate_results.csv")
    plot_metric(rows, args.output_dir / "psnr_vs_bpp.png", "psnr", "Average PSNR vs BPP", "PSNR (dB)")
    plot_metric(rows, args.output_dir / "ssim_vs_bpp.png", "ssim", "Average SSIM vs BPP", "SSIM")

    print(f"Processed {len(images)} image(s): {', '.join(path.name for path in images)}")
    print(f"Saved: {args.output_dir / 'bitrate_results.csv'}")
    print(f"Saved: {args.output_dir / 'psnr_vs_bpp.png'}")
    print(f"Saved: {args.output_dir / 'ssim_vs_bpp.png'}")


if __name__ == "__main__":
    main()
