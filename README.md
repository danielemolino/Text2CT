<div align="center">

# From Alignment to Synthesis Contrastive Volumetric Grounding for Text-to-CT Generation

<p>
  <a href="[https://bmvc2026.org/](https://bmvc2026.bmva.org)"><img src="https://img.shields.io/badge/BMVC-2026-4b2e83?style=for-the-badge&labelColor=1a1a1a" alt="BMVC 2026"></a>
  <a href="https://arxiv.org/abs/2506.00633"><img src="https://img.shields.io/badge/arXiv-2506.00633-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a1a" alt="arXiv"></a>
  <a href="https://huggingface.co/dmolino/text2ct-weights"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-text2ct-ffcc4d?style=for-the-badge&labelColor=1a1a1a" alt="Weights"></a>
  <a href="https://huggingface.co/datasets/dmolino/CT-RATE_Generated_Scans"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-1k%20synthetic%20CTs-ffcc4d?style=for-the-badge&labelColor=1a1a1a" alt="Dataset"></a>
</p>

**Daniele Molino**<sup>1</sup> · **Camillo Maria Caruso**<sup>1</sup> · **Filippo Ruffini**<sup>1,2</sup> · **Valerio Guarrasi**<sup>3</sup> · **Paolo Soda**<sup>1,2</sup>

<sub><sup>1</sup> Università Campus Bio-Medico di Roma &nbsp;·&nbsp; <sup>2</sup> Umeå University &nbsp;·&nbsp; <sup>3</sup> UniCamillus</sub>

<p>
  <a href="https://arxiv.org/abs/2506.00633"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="#-model-overview"><b>Method</b></a> &nbsp;·&nbsp;
  <a href="#-synthetic-dataset"><b>Dataset</b></a> &nbsp;·&nbsp;
  <a href="#-getting-started"><b>Getting started</b></a> &nbsp;·&nbsp;
  <a href="#-citation"><b>Citation</b></a>
</p>

</div>

---

> [!IMPORTANT]
> ### 🎉 Accepted at BMVC 2026
>
> This work has been **accepted at BMVC 2026** in a substantially extended form. The paper now asks a different question: is the bottleneck for semantic controllability in Text-to-CT the *richness of the text encoder*, or the *quality of 3D vision-language grounding*? We show it is the latter, and introduce a generation-oriented 3D-CLIP encoder trained with **text-level structured hard negatives**.
>
> 📄 **Updated paper:** [arXiv:2506.00633](https://arxiv.org/abs/2506.00633) (v3)
> 🚧 **Code:** what you find below is the release accompanying the original preprint. The BMVC version — hard-negative contrastive training and updated checkpoints — will land here *<!-- edit: e.g. by March 2026 -->*. ⭐ the repo to get notified.

---

![model](https://github.com/cosbidev/Text2CT/blob/main/model.png)

## 🧠 Model Overview

Generating semantically controllable 3D CT volumes from radiology reports requires more than a rich text encoder — it requires vision-language alignment **grounded in volumetric space**. Existing Text-to-CT approaches condition generation on encoders pretrained with language-only or 2D vision-language objectives: linguistically expressive, but volumetrically blind.

Our framework combines:

- **A generation-oriented 3D-CLIP encoder**, trained contrastively on paired CT volumes and radiology reports with **structured hard negatives operating exclusively at the text level**. Because confusable reports cost nothing in 3D memory, this raises contrastive difficulty without the small-batch ceiling that constrains volumetric encoders. Beyond conditioning, the encoder reaches state-of-the-art zero-shot pathology classification and volumetric retrieval.
- **A volumetric VAE** compressing CT scans into a low-dimensional latent space.
- **A fully end-to-end latent diffusion model** operating directly in 3D latent space, with cross-attention conditioning — no cascaded super-resolution, and therefore none of the spatial artifacts and cross-slice inconsistencies it introduces.

Evaluated on CT-RATE across 18 pathological conditions, the method reaches state-of-the-art performance on both image fidelity and **factual correctness**, at lower inference time and GPU memory than competing approaches. A recurring finding of the paper: fidelity metrics such as FID can look healthy while the model quietly ignores the conditioning report — only task-oriented metrics expose it.

---

## 📦 Synthetic Dataset

We release **1,000 synthetic chest CT scans** generated with our model for the [VLM3D Challenge](https://vlm3dchallenge.com).

➡️ [Synthetic Text-to-CT Dataset on Hugging Face](https://huggingface.co/datasets/dmolino/CT-RATE_Generated_Scans)

---

## 🚀 Getting Started

> [!NOTE]
> The instructions below reproduce the **preprint** version of the model. They will be updated alongside the BMVC code release.

### Environment

Python 3.10.8

```bash
pip install -r requirements.txt
```

### Weights

Download from [🤗 dmolino/text2ct-weights](https://huggingface.co/dmolino/text2ct-weights) and place in `models/`:

| File | Component |
| --- | --- |
| `autoencoder_epoch273.pt` | Volumetric VAE |
| `unet_rflow_200ep.pt` | Diffusion UNet |
| `CLIP3D_Finding_Impression_30ep.pt` | 3D-CLIP text/vision encoder |

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dmolino/text2ct-weights",
    repo_type="model",
    local_dir="your_local_path",
)
```

Then set the paths in the configs:

| Config field | Points to |
| --- | --- |
| `trained_autoencoder_path` | autoencoder |
| `existing_ckpt_filepath` / `model_filename` | unet |
| `clip_weights` | clip |

### Data

We use the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset.

```bash
python scripts/download_ctrate.py     # pull volumes from HF
python scripts/preprocess_ctrate.py   # reorient to RAS, clip HU, resample to fixed spacing/shape
```

After download and preprocessing, make sure:

- `dataset/` contains the CT volumes
- `data/train_data_volumes.json` and `data/validation_data_volumes.json` list volumes with relative paths (e.g. `dataset/train/...`)
- `data/train_reports.csv` and `data/validation_reports.csv` contain the reports (`VolumeName`, `Findings_EN`, `Impressions_EN`)

### Precomputing embeddings

Recommended — it speeds up training considerably.

**1. VAE latent embeddings (CT)**

```bash
python scripts/diff_model_create_training_data.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1 \
  --index 0
```

Key fields in `environment_diff_model_train.json`:

- `data_base_dir` → `dataset`
- `embedding_base_dir` → output folder for latents (e.g. `./embeddings`)
- `trained_autoencoder_path` → `./models/autoencoder_epoch273.pt`

**2. Report embeddings (3D-CLIP)**

```bash
python scripts/save_embeddings_ctrate.py \
  --train_json data/train_data_volumes.json \
  --val_json data/validation_data_volumes.json \
  --train_reports data/train_reports.csv \
  --val_reports data/validation_reports.csv \
  --data_base_dir dataset \
  --embedding_base_dir ./embeddings \
  --clip_weights ./models/CLIP3D_Finding_Impression_30ep.pt \
  --report_encoder_model xgem_3D
```

### Training

```bash
python scripts/diff_model_train.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1
```

Use `existing_ckpt_filepath` to resume from your own checkpoint.

### Inference

```bash
python scripts/diff_model_infer.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_eval.json \
  --num_gpus 1 \
  --index 0 \
  --resize 512
```

Outputs are written to the `output_dir` set in `environment_diff_model_eval.json`.

### Demo — generate from your own report

```bash
python scripts/diff_model_demo.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_eval.json \
  --num_gpus 1
```

Edit `example_report` inside the script. Output: `predictions/demo.nii.gz`.

### Script reference

| Script | What it does |
| --- | --- |
| `scripts/download_ctrate.py` | Download CT-RATE volumes from HF |
| `scripts/preprocess_ctrate.py` | Reorient / clip / resample to fixed spacing and shape |
| `scripts/save_embeddings_ctrate.py` | Encode reports with 3D-CLIP, save impressions as `.npy` |
| `scripts/diff_model_create_training_data.py` | Extract VAE latent embeddings for CT volumes |
| `scripts/diff_model_train.py` | Train the diffusion UNet |
| `scripts/diff_model_infer.py` | Batch inference over data lists |
| `scripts/diff_model_demo.py` | One-off generation from a provided report |

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{molino2026alignment,
  title     = {From Alignment to Synthesis: Contrastive Volumetric Grounding for Text-to-CT Generation},
  author    = {Molino, Daniele and Caruso, Camillo Maria and Ruffini, Filippo and Guarrasi, Valerio and Soda, Paolo},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

---

## 📬 Contact

Questions or collaborations — **Daniele Molino**, [daniele.molino@unicampus.it](mailto:daniele.molino@unicampus.it)

## 🙏 Acknowledgements

This repository builds on:

- [MAISI tutorials (MONAI)](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi)
- [XGeM](https://github.com/cosbidev/XGeM)
