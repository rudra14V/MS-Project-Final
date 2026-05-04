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
