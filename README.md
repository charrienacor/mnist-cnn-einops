# MNIST — 3-Layer CNN built entirely with `einops` / `einsum`

A 3-layer convolutional network for MNIST digit classification in which **every**
core operation is implemented by hand with [`einops`](https://github.com/arogozhnikov/einops):
`rearrange`, `reduce`, and `einsum`. No `nn.Conv2d`, `nn.BatchNorm2d`,
`nn.MaxPool2d`, `nn.Linear`, or `.view()` anywhere in the model.

![test accuracy](https://img.shields.io/badge/test%20split%20accuracy-98.27%25-green)

## Architecture

| Stage | Operation | Implementation |
|---|---|---|
| Conv 1 | 1 → 32 ch, 3×3, pad 1 | `einsum` over kernel positions: `"b c oh ow, o c -> b o oh ow"` |
| BN 1 | 32 ch | channel stats via `reduce(..., "mean"/"std")` |
| Pool 1 | 2×2 max | `rearrange` into windows, then `reduce(..., "max")` |
| Conv 2 | 32 → 64 ch, 3×3, pad 1 | same einsum contract |
| BN 2 | 64 ch | einops reduce |
| Pool 2 | 2×2 max | fold + reduce |
| Conv 3 | 64 → 128 ch, 3×3 | same einsum contract |
| BN 3 | 128 ch | einops reduce |
| FC 1 | 6272 → 256 | 2-way `einsum`: `"bo, oi -> bi"` |
| FC 2 | 256 → 10 | 2-way `einsum` |

ReLU + dropout between stages. **Total: 1,701,578 parameters.**

The custom `Conv2dEinops` is verified in-notebook against `nn.Conv2d`
(max abs diff ≈ 1e-7).

## Data & training

- **Data:** MNIST 10k test set used as the data pool, split (seeded) into
  **7,000 train / 3,000 test** so "test split accuracy" is honest.
- **Training:** 5 epochs, Adam (lr 1e-3), batch size 128, CPU (~27 s).
- **Result: test-split accuracy 98.27 % (2948/3000).**

## Notebook contents

`mnist_cnn_einops.ipynb` is fully executed and shows:

1. Imports & seed setup
2. Data split
3. Einops building blocks (`Conv2dEinops`, `BatchNorm2dEinops`, `MaxPool2dEinops`,
   `FlattenEinops`, `LinearEinops`)
4. Model definition
5. Training loop (5 epochs)
6. Test-split evaluation
7. Train loss / accuracy curves
8. **4×4 grid of 16 sampled images with ground truth vs. prediction**

## Run it

```bash
pip install torch torchvision einops matplotlib
jupyter notebook mnist_cnn_einops.ipynb
```
