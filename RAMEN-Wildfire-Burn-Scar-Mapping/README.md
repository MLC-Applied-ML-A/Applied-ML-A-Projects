# Wildfire Burn-Scar Mapping with RAMEN

Applied ML A, CS197 study group. Active members: **Daniel Odeh** and **Nwajiunor Edgar-Odurombi** (both worked on every stage together).

We fine-tuned RAMEN, a pretrained Earth observation foundation model (ViT-Base encoder), with a UPerNet decoder to label every pixel of a satellite image as **Not burned** or **Burn scar**. This maps land that has already burned. It does not detect live flames or predict future fires.

## Contents

| Path | What it is |
|---|---|
| `Applied_ML_A_Finetuning_RAMEN.ipynb` | Colab notebook: setup, training command, evaluation and figures |
| `figures/` | Seven test samples: satellite image, ground truth, prediction |
| `slides/` | Presentation deck |

Colab version: https://colab.research.google.com/drive/1nOMboAHWRYfOJOVZtR9NrKtd6SZxkZnL?usp=sharing

## Data

[HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) (IBM-NASA Geospatial, Hugging Face). 512 x 512 tiles, six bands (B2, B3, B4, B8A, B11, B12). About 88.9% of pixels are not burned and 11.1% are burn scar, so accuracy alone is misleading.

## Setup

Run through the PANGAEA benchmark inside the [RAMEN repository](https://github.com/nicolashoudre/RAMEN) (`pangaea-bench`), on a Colab Tesla T4.

| Setting | Value |
|---|---|
| Encoder / decoder | RAMEN ViT-Base, fine-tuned (`finetune=True`) / UPerNet |
| Resolution | `input_res=30`, `res=480`, `input_size=512` |
| Epochs | 80 |
| Batch size | 2 |
| Optimiser | AdamW, lr 0.0001, weight decay 0.05 |
| Precision | FP16 mixed precision |
| Loss | Cross-entropy |
| Checkpoint | Best validation mIoU |

Two fixes were needed on Colab's newer PyTorch: `torch.cuda.amp.GradScaler` changed to `torch.amp.GradScaler` in `pangaea/engine/trainer.py`, and `hydra-core` installed separately (PANGAEA was installed with `--no-deps` to protect Colab's torch).

## Results (test set, 264 batches)

| Metric (%) | Not burned | Burn scar | Mean |
|---|---:|---:|---:|
| IoU | 97.017 | 73.732 | 85.374 |
| F1 | 98.486 | 84.880 | 91.683 |
| Precision | 98.654 | 83.459 | 91.056 |
| Recall | 98.318 | 86.351 | 92.334 |

Mean accuracy: 97.248%.

Burn-scar recall is higher than precision, so the model catches most burned pixels but flags some false positives. Predicted shapes are smoother than the ground truth, which fits the coarse 480 m feature grid.

## Next steps

Handle class imbalance (weighted or Dice loss), add data augmentation, try other model configurations, and evaluate on other regions.

## References

- Houdré et al. (2026). RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation. CVPR 2026. arXiv:2512.05025
- IBM-NASA Geospatial (2023). HLS Burn Scars [Dataset]. Hugging Face. doi:10.57967/hf/0956
- Marsocci et al. (2024). PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models. arXiv:2412.04204
- Xiao et al. (2018). Unified Perceptual Parsing for Scene Understanding. ECCV
