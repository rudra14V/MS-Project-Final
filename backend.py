"""FastAPI backend for APUBT3-NUP image compression."""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from apubt3_nup import (
    decode_apubt3_array,
    encode_apubt3,
    export_jpeg as apubt3_export_jpeg,
)
from generate_bitrate_graphs import (
    bpp as compute_bpp,
    psnr as compute_psnr,
    save_baseline_jpeg,
    ssim as compute_ssim,
)

app = FastAPI(title="APUBT3-NUP Compression API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_bool(value: str) -> bool:
    return str(value).lower() in ("true", "1", "on", "yes")


def _rgb_to_b64(rgb: np.ndarray, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _load_rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


@app.get("/")
def health():
    return {"status": "ok", "service": "APUBT3-NUP API"}


@app.post("/api/encode")
async def encode_endpoint(
    file: UploadFile = File(...),
    quality: int = Form(85),
    target_ratio: float = Form(0.60),
    preserve_residual: str = Form("false"),
):
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(400, "Please upload a JPEG image (.jpg or .jpeg)")
    if not 0.05 <= target_ratio < 1.0:
        raise HTTPException(400, "target_ratio must be between 0.05 and 0.99")

    raw = await file.read()
    original_rgb = _load_rgb(raw)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.jpg"
        input_path.write_bytes(raw)
        output_path = tmp / "compressed.apubt3"

        try:
            metadata = encode_apubt3(
                input_path,
                output_path,
                quality=quality,
                preserve_residual=_to_bool(preserve_residual),
                target_ratio=target_ratio,
            )
        except Exception as exc:
            raise HTTPException(500, f"Encoding failed: {exc}")

        compressed_bytes = output_path.read_bytes()
        reconstructed_rgb, _ = decode_apubt3_array(output_path)

    metadata["psnr"] = round(compute_psnr(original_rgb, reconstructed_rgb), 4)
    metadata["ssim"] = round(compute_ssim(original_rgb, reconstructed_rgb), 6)
    metadata["reconstructed_b64"] = _rgb_to_b64(reconstructed_rgb)
    metadata["file_b64"] = base64.b64encode(compressed_bytes).decode()
    return metadata


@app.post("/api/export-jpeg")
async def export_jpeg_endpoint(
    file: UploadFile = File(...),
    quality: int = Form(85),
    target_ratio: float = Form(0.60),
    preserve_residual: str = Form("false"),
):
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(400, "Please upload a JPEG image (.jpg or .jpeg)")
    if not 0.05 <= target_ratio < 1.0:
        raise HTTPException(400, "target_ratio must be between 0.05 and 0.99")

    raw = await file.read()
    original_rgb = _load_rgb(raw)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.jpg"
        input_path.write_bytes(raw)
        output_path = tmp / "compressed.jpg"

        try:
            metadata = apubt3_export_jpeg(
                input_path,
                output_path,
                quality=quality,
                preserve_residual=_to_bool(preserve_residual),
                target_ratio=target_ratio,
            )
        except Exception as exc:
            raise HTTPException(500, f"Export failed: {exc}")

        compressed_jpeg_bytes = output_path.read_bytes()
        compressed_rgb = np.asarray(Image.open(output_path).convert("RGB"))

    metadata["psnr"] = round(compute_psnr(original_rgb, compressed_rgb), 4)
    metadata["ssim"] = round(compute_ssim(original_rgb, compressed_rgb), 6)
    metadata["image_b64"] = base64.b64encode(compressed_jpeg_bytes).decode()
    return metadata


@app.post("/api/decode")
async def decode_endpoint(file: UploadFile = File(...)):
    raw = await file.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "compressed.apubt3"
        input_path.write_bytes(raw)

        try:
            rgb, metadata = decode_apubt3_array(input_path)
        except Exception as exc:
            raise HTTPException(400, f"Failed to decode file: {exc}")

    metadata["image_b64"] = _rgb_to_b64(rgb)
    return metadata


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    quality: int = Form(85),
    ratios: str = Form("0.30,0.40,0.50,0.60,0.70,0.80"),
):
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(400, "Please upload a JPEG image (.jpg or .jpeg)")

    try:
        ratio_list = [float(r.strip()) for r in ratios.split(",")]
        if not all(0.05 <= r < 1.0 for r in ratio_list):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "ratios must be comma-separated floats between 0.05 and 0.99")

    raw = await file.read()
    original_rgb = _load_rgb(raw)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.jpg"
        input_path.write_bytes(raw)

        for ratio in ratio_list:
            apubt3_out = tmp / f"apubt3_{ratio:.2f}.apubt3"
            jpeg_out = tmp / f"jpeg_{ratio:.2f}.jpg"

            try:
                apubt3_meta = encode_apubt3(input_path, apubt3_out, quality=quality, target_ratio=ratio)
                apubt3_rgb, _ = decode_apubt3_array(apubt3_out)
                jpeg_meta = save_baseline_jpeg(original_rgb, input_path, jpeg_out, quality, ratio)
                jpeg_rgb = np.asarray(Image.open(jpeg_out).convert("RGB"))
            except Exception as exc:
                raise HTTPException(500, f"Analysis failed at ratio {ratio}: {exc}")

            for method, out_path, recon_rgb, meta in [
                ("APUBT3-NUP", apubt3_out, apubt3_rgb, apubt3_meta),
                ("JPEG", jpeg_out, jpeg_rgb, jpeg_meta),
            ]:
                rows.append({
                    "method": method,
                    "target_ratio": ratio,
                    "bpp": round(compute_bpp(out_path, original_rgb.shape), 4),
                    "psnr": round(compute_psnr(original_rgb, recon_rgb), 4),
                    "ssim": round(compute_ssim(original_rgb, recon_rgb), 6),
                    "compressed_bytes": meta["compressed_bytes"],
                    "original_bytes": meta["original_bytes"],
                })

    return {"rows": rows}
