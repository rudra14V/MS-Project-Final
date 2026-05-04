# APUBT3-NUP Color Image Compression

This project implements the core idea from `MS paper.pdf`: a JPEG-like image
compression model based on non-uniform partitioning and APUBT3 transform
coding.

## What Is Implemented

- ITANRP texture classification using least-squares plane fitting on 16x16 luma
  blocks.
- Non-uniform block layout: smooth areas stay 16x16, detailed areas split into
  four 8x8 blocks.
- APUBT3-style transform coding for both 8x8 and 16x16 blocks.
- Adaptive scalar quantization with separate coefficients for detailed and
  smooth regions.
- Color-only RGB input/output through YCrCb processing.
- Custom `.apubt3` compressed bitstream for paper-style evaluation.
- Optional JPEG export for convenient viewing.

## Usage

Encode your JPEG image into the paper-style compressed bitstream:

```powershell
python apubt3_nup.py encode input.jpg compressed_image.apubt3 --quality 85
```

Decode the compressed bitstream to a viewable image:

```powershell
python apubt3_nup.py decode compressed_image.apubt3 reconstructed.png
```

Use `--target-ratio` to control transform quantization strength:

```powershell
python apubt3_nup.py encode input.jpg compressed_image.apubt3 --target-ratio 0.50
```

For convenient JPEG-only viewing, export a JPEG:

```powershell
python apubt3_nup.py export-jpeg input.jpg compressed_image.jpg --quality 85
```

## Notes

The paper's entropy coding tables are JPEG-specific. This implementation stores
the APUBT3-NUP quantized coefficients in a custom `.apubt3` archive so metrics
can be calculated from the proposed codec before any extra JPEG saving loss.

## Code Flow (High Level)

1. CLI parses a subcommand: `encode`, `decode`, or `export-jpeg`.
2. Input JPEG is loaded (for `encode`/`export-jpeg`) and converted to RGB.
3. RGB is converted to YCrCb and padded to a multiple of 16.
4. ITANRP decides per 16x16 tile whether to keep 16x16 or split into 8x8.
5. Each block is APUBT3-transformed and scalar-quantized.
6. Coefficients are zigzag-scanned and run-length encoded.
7. Data is packed into a compressed NumPy archive for `encode`, or decoded
   back to RGB and optionally saved as JPEG for `export-jpeg`.
8. `decode` reads the archive, reconstructs blocks, and writes the image.

## Function-Wise Explanation

### Transform and Basis Helpers

- `apubt3_matrix(size)`: Returns the transform matrix used for APUBT3 at 8x8 or
  16x16. This implementation uses a stable U-system degree-3 basis for both
  sizes.
- `_legendre_u_degree3_basis(n)`: Builds and orthogonalizes a discrete
  degree-3-like basis used to approximate APUBT3.
- `_all_phase_matrix(core, iterations)`: Prototype APDF iteration (kept for
  completeness; not used in the main pipeline).

### Image Prep and Color Space

- `pad_to_multiple(image, multiple=16)`: Pads an image to a multiple of 16 using
  reflected borders and returns the padded image and original shape.
- `rgb_to_ycbcr(rgb)`: Converts RGB to YCrCb using OpenCV for luminance-based
  partitioning.
- `ycbcr_to_rgb(ycbcr)`: Clips and converts back to RGB for output.

### ITANRP Partitioning

- `_fit_plane_rmse(block)`: Fits a plane to a block and returns RMSE; low RMSE
  means the block is smooth.
- `itanrp_partition(luma, epsilon)`: Applies the plane test per 16x16 luma tile.
  Smooth tiles stay 16x16; detailed tiles split into four 8x8 blocks.

### Scan and Entropy Helpers

- `zigzag_indices(size)`: Returns zigzag order for coefficients (JPEG-style).
- `rle_encode(values)`: Run-length encodes a flat coefficient sequence.
- `rle_decode(pairs)`: Decodes RLE pairs back to a flat sequence.

### Block Transform Coding

- `_encode_block(block, matrix, q)`: Centers a block, applies the APUBT3
  transform, and scalar-quantizes the coefficients.
- `_decode_block(values, matrix, q)`: Dequantizes and inverse-transforms a
  block.
- `adaptive_q2(q1, block_count, detail_count, mu=100.0)`: Derives the smoother
  16x16 quantizer from the 8x8 quantizer based on texture prevalence.

### Payload Build and Rebuild

- `_build_payload(...)`: Core encoder. Loads input, partitions blocks, encodes
  coefficients, and optionally computes a residual correction layer. Returns
  metadata, block payloads, and optional residual.
- `_rebuild_from_payload(payload_blocks, metadata, residual=None)`: Core
  decoder. Reconstructs YCrCb blocks from quantized coefficients, converts to
  RGB, and applies residual if present.

### Archive Packing

- `_payload_to_arrays(metadata, payload_blocks, residual)`: Packs metadata and
  RLE-trimmed coefficients into NumPy arrays for storage.
- `encode_apubt3(...)`: Iteratively increases quantization until the target
  compression ratio is reached, then writes the `.apubt3` archive.

### Archive Decode and JPEG Export

- `decode_apubt3_array(input_path)`: Loads the archive, reconstructs the
  coefficient payload, and returns the decoded RGB image and metadata.
- `decode_apubt3(input_path, output_path)`: Writes the decoded RGB image.
- `export_jpeg(...)`: Runs the transform pipeline and saves the result as JPEG
  at the best quality that fits the target size.
- `_jpeg_bytes(rgb, quality)`: Helper that encodes to JPEG in memory.
- `_write_target_jpeg(...)`: Searches for the highest JPEG quality that meets
  the target byte size.

### CLI Entrypoint

- `main()`: Parses CLI args, validates `--target-ratio`, and dispatches to
  `encode_apubt3`, `decode_apubt3`, or `export_jpeg`.

## Bitrate Graphs

Generate paper-style bitrate curves:

```powershell
python generate_bitrate_graphs.py
```

The script writes:

- `bitrate_graphs/psnr_vs_bpp.png`
- `bitrate_graphs/ssim_vs_bpp.png`
- `bitrate_graphs/bitrate_results.csv`

It compares APUBT3-NUP custom bitstreams with baseline JPEG across several
compressed/original file-size ratios.
