# Paper Analysis

Paper: **A New Image Compression Algorithm Based on Non-Uniform Partition and
U-System**, IEEE Transactions on Multimedia, 2021.

## Main Problem

Baseline JPEG uses fixed 8x8 blocks, DCT, and a static quantization table. At
low bit rates this often creates visible blocking artifacts, especially in
smooth areas. The paper improves this by changing the partitioning, transform
core, and quantization strategy.

## Implemented Algorithm

1. **ITANRP partitioning**
   The image is first split into 16x16 tiles. For every tile, a least-squares
   plane `z = ax + by + c` is fitted to the luma pixels. The RMSE decides texture
   complexity:
   - low RMSE: keep one 16x16 block
   - high RMSE: split into four 8x8 blocks

2. **APUBT3-style transform coding**
   Each block is transformed before quantization. The printed 8x8 APUBT3 matrix
   from the paper is included in code as a reference. For stable reconstruction,
   the implementation uses an orthogonalized discrete U-system degree-3 basis
   for both 8x8 and 16x16 blocks.

3. **Adaptive quantization**
   Detailed 8x8 blocks use `q1`; smoother 16x16 blocks use `q2`, calculated from
   the paper's ratio:

   `q2 / q1 = mu * (3Lsize - 2L1) / (2Lsize)`

   where `Lsize` is the number of 16x16 tiles and `L1` is the number of detailed
   tiles.

4. **Color-only processing**
   The paper experiments mostly use grayscale images. This project extends the
   workflow to color images by converting RGB to YCrCb, partitioning by luma, and
   coding all three channels. Output is always reconstructed as RGB color.

5. **JPEG output**
   The compressed result is saved directly as a standard `.jpg` image. The
   optional `--preserve-residual` flag applies an RGB correction before saving,
   which helps the JPEG stay visually closer to the input.

## Difference From A Production Codec

The paper mentions JPEG-like entropy coding. This project keeps the paper-style
partition, transform, and adaptive quantization stages, then saves the final
compressed reconstruction as a normal JPEG file for easy viewing.
