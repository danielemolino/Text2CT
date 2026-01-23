# Text-to-CT Generation via 3D Latent Diffusion Model

**[Text-to-CT Generation via 3D Latent Diffusion Model with Contrastive Vision-Language Pretraining](https://arxiv.org/abs/2506.00633)**  
Molino, D., Caruso, C. M., Ruffini, F., Soda, P., Guarrasi, V. (2025)

![model](https://github.com/cosbidev/Text2CT/blob/main/model.png)

---

## 🧠 Model Overview
Our approach combines:
- A **3D CLIP-style encoder** for vision-language alignment between CT volumes and radiology reports.
- A **volumetric VAE** for latent compression of 3D CT data.
- A **Latent Diffusion Model** with cross-attention conditioning for controllable text-to-CT generation.  
This design enables direct synthesis of anatomically consistent, semantically faithful, and high-resolution CT volumes from textual descriptions.

---

## 📦 Synthetic Dataset
We release **1,000 synthetic chest CT scans** generated with our model for the [VLM3D Challenge](https://vlm3dchallenge.com).  
➡️ Available on Hugging Face: [Synthetic Text-to-CT Dataset](https://huggingface.co/datasets/dmolino/CT-RATE_Generated_Scans)  

---

## 📜 Paper
- Preprint: [arXiv:2506.00633](https://arxiv.org/abs/2506.00633)

---

## 🚧 Code Release

### Environment
- Python 3.10.8
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Weights (place in `models/`)
- `autoencoder_epoch273.pt`
- `unet_rflow_200ep.pt`
- `CLIP3D_Finding_Impression_30ep.pt`

You can download them from Hugging Face at [Weights](https://huggingface.co/dmolino/text2ct-weights):
```python
from huggingface_hub import snapshot_download

repo_id = "dmolino/text2ct-weights"

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir="your_local_path" 
)
```
Set these paths in the configs:
- `trained_autoencoder_path` -> `autoencoder`
- `existing_ckpt_filepath` / `model_filename` -> `unet`
- `clip_weights` -> `clip`

### Script quick reference
- `scripts/download_ctrate.py`: download CT-RATE volumes from HF.
- `scripts/preprocess_ctrate.py`: reorient/clip/resample CT-RATE to fixed spacing/shape.
- `scripts/save_embeddings_ctrate.py`: encode reports with CLIP3D and save impressions as npy.
- `scripts/diff_model_create_training_data.py`: extract VAE latent embeddings for CT volumes.
- `scripts/diff_model_train.py`: train diffusion UNet.
- `scripts/diff_model_infer.py`: batch inference over data lists.
- `scripts/diff_model_demo.py`: one-off generation from a provided report (no precomputed impressions).
---

### Data
We use the CT-RATE dataset (Hugging Face). Helpers provided:
- Download: `scripts/download_ctrate.py` (pull volumes from HF).
- Preprocess: `scripts/preprocess_ctrate.py` (reorient to RAS, clip HU, resample to fixed spacing/shape).

After download/preprocess, ensure:
- `dataset/` contains the CT volumes.
- `data/train_data_volumes.json` and `data/validation_data_volumes.json` list volumes with relative paths (e.g., `dataset/train/...`).
- `data/train_reports.csv` and `data/validation_reports.csv` contain the text reports (`VolumeName`, `Findings_EN`, `Impressions_EN`).

### Embeddings (recommended to speed up training)
1) **VAE latent embeddings (CT)** – `scripts/diff_model_create_training_data.py`:
```bash
python scripts/diff_model_create_training_data.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1 \
  --index 0
```
Key fields in `environment_diff_model_train.json`:
- `data_base_dir`: set to `dataset`
- `embedding_base_dir`: output folder for latents (e.g., `./embeddings`)
- `trained_autoencoder_path`: `./models/autoencoder_epoch273.pt`

2) **Report embeddings (CLIP3D)** – `scripts/save_embeddings_ctrate.py`:
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
Train the diffusion UNet – `scripts/diff_model_train.py`:
```bash
python scripts/diff_model_train.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1
```
Weights expected in `models/`:
- `autoencoder_epoch273.pt`
- `unet_rflow_200ep.pt` (or your own checkpoint via `existing_ckpt_filepath`)

### Inference
Run inference over a data list – `scripts/diff_model_infer.py`:
```bash
python scripts/diff_model_infer.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_eval.json \
  --num_gpus 1 \
  --index 0 \
  --resize 512
```
Outputs go to `output_dir` set in `environment_diff_model_eval.json`.

### Demo (quick test from a report)
Generate a CT volume from a custom report (no precomputed impressions) – `scripts/diff_model_demo.py`:
```bash
python scripts/diff_model_demo.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_eval.json \
  --num_gpus 1
```
Edit `example_report` inside the script to your text. Output: `predictions/demo.nii.gz`.

## 📬 Contact
For questions or collaborations:  
**Daniele Molino** – [daniele.molino@unicampus.it](mailto:daniele.molino@unicampus.it)

---

## Acknowledgements
This repository is heavily based on:
- MAISI tutorials (MONAI): https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi
- XGeM: https://github.com/cosbidev/XGeM
