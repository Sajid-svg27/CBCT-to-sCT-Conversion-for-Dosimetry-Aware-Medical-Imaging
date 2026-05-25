

````markdown
# CBCT-to-Synthetic CT Generation for Adaptive Radiotherapy

This project implements a CBCT-to-synthetic CT generation pipeline using the **SynthRAD2025 Task 2** dataset. The goal is to convert cone-beam CT images into CT-like synthetic CT images with improved HU accuracy for adaptive radiotherapy applications.

The project was developed as a medical imaging / medical physics AI portfolio project, focusing on practical CBCT artifact correction, HU restoration, region-wise evaluation, and exploratory uncertainty estimation.

---

## Project Motivation

Cone-beam CT is widely used in image-guided and adaptive radiotherapy because it provides patient anatomy at the treatment position. However, CBCT images suffer from several limitations:

- scatter artifacts,
- noise,
- shading/cupping artifacts,
- inaccurate HU values,
- reduced soft-tissue contrast.

These limitations make raw CBCT less reliable for quantitative radiotherapy tasks such as dose calculation or adaptive treatment planning.

Synthetic CT generation aims to learn a mapping:

```text
CBCT image → CT-like synthetic CT image
````

so that the output has improved anatomical and HU consistency compared with raw CBCT.

---

## Dataset

The project uses the **SynthRAD2025 Task 2** dataset.

Input:

```text
CBCT
```

Target:

```text
Planning CT
```

Anatomical regions:

```text
AB  = abdomen
HN  = head and neck
TH  = thorax
```

Each patient folder contains:

```text
cbct.mha
ct.mha
mask.mha
```

Expected dataset structure:

```text
data/
└── Task2/
    ├── AB/
    │   ├── 2ABA002/
    │   │   ├── cbct.mha
    │   │   ├── ct.mha
    │   │   └── mask.mha
    │   └── ...
    ├── HN/
    └── TH/
```

The dataset is **not included** in this repository.

---

## Project Pipeline

The complete pipeline is:

```text
CBCT/CT/mask loading
        ↓
HU clipping and normalization
        ↓
center crop / padding to 320 × 320
        ↓
2.5D slice-stack creation
        ↓
2.5D U-Net++ model
        ↓
synthetic CT prediction
        ↓
HU-based evaluation
        ↓
region-wise error analysis
        ↓
Monte Carlo dropout uncertainty
```

---

## Preprocessing

Images are loaded using **SimpleITK**.

The `.mha` files are read in array format:

```text
[z, y, x]
```

All images are clipped to a fixed HU range:

```text
min HU = -1000
max HU = 1600
```

Then normalized to:

```text
[-1, 1]
```

using:

```text
x_norm = 2 × (x - minHU) / (maxHU - minHU) - 1
```

Because patient image sizes vary, each slice is center-cropped or padded to:

```text
320 × 320
```

The original voxel spacing was consistent across the dataset examples:

```text
1.0 × 1.0 × 3.0 mm
```

---

## 2.5D Dataset Strategy

Instead of using only one CBCT slice, the model receives neighboring slices as multiple input channels.

For a center slice `z`, the input is:

```text
CBCT[z-3], CBCT[z-2], CBCT[z-1], CBCT[z], CBCT[z+1], CBCT[z+2], CBCT[z+3]
```

The target is:

```text
CT[z]
```

So the model input/output shapes are:

```text
Input:  [7, 320, 320]
Target: [1, 320, 320]
```

This gives the model local 3D anatomical context while still keeping the memory cost closer to 2D training.

---

## Model Architecture

The final model uses:

```text
2.5D U-Net++ with ResNet-34 encoder
```

implemented using:

```text
segmentation_models_pytorch
```

Although the library is commonly used for segmentation, this project uses the model for **image-to-image regression**.

Model configuration:

| Component       |                Setting |
| --------------- | ---------------------: |
| Architecture    |                U-Net++ |
| Encoder         |              ResNet-34 |
| Encoder weights |               ImageNet |
| Input channels  |                      7 |
| Output channels |                      1 |
| Image size      |              320 × 320 |
| Dropout         |                    0.2 |
| Task            | CBCT-to-sCT regression |

---

## Loss Function

The final training loss is:

```text
Total Loss = L1 Loss + 0.2 × SSIM Loss + 0.05 × Gradient Loss
```

Purpose of each term:

| Loss term     | Purpose                                     |
| ------------- | ------------------------------------------- |
| L1 loss       | improves voxel-wise HU accuracy             |
| SSIM loss     | encourages anatomical structural similarity |
| Gradient loss | preserves edges and tissue boundaries       |

A region-weighted loss was also tested as an ablation study, but it did not improve overall performance.

---

## Computational Constraints

This project was developed under standard **Google Colab T4 GPU** limitations.

The available GPU memory, session duration, and runtime stability constrained the scale of training. Therefore, the final Colab-based model used a strong **single-fold subset training strategy** instead of full challenge-style multi-fold ensembling or full-dataset training.

Final Colab training setup:

| Setting             |     Value |
| ------------------- | --------: |
| GPU                 | NVIDIA T4 |
| Training patients   |        80 |
| Validation patients |        20 |
| Epochs              |        25 |
| Batch size          |         2 |
| Image size          | 320 × 320 |
| Input slices        |         7 |
| Best epoch          |        12 |

The code supports resume training using saved checkpoints, which makes it suitable for future scaling on RTX 4090, cloud GPU, or cluster resources.

---

## Final Deterministic Results

The final deterministic model was evaluated by comparing raw CBCT and predicted synthetic CT against the ground-truth CT.

| Metric    | Raw CBCT | Final sCT |
| --------- | -------: | --------: |
| MAE [HU]  |   216.97 |     88.84 |
| RMSE [HU] |   271.23 |    170.50 |
| PSNR      |    19.99 |     24.08 |
| SSIM      |    0.654 |     0.330 |

The final model reduced overall MAE by approximately:

```text
59.05%
```

compared with raw CBCT.

---

## Region-wise Results

Region-wise MAE improvements:

| Region      | Raw CBCT MAE [HU] | Final sCT MAE [HU] | Improvement |
| ----------- | ----------------: | -----------------: | ----------: |
| Body        |            216.97 |              88.84 |      59.05% |
| Bone        |            391.34 |             213.66 |      45.40% |
| Soft tissue |            210.86 |              60.59 |      71.26% |
| Air/lung    |            153.81 |             186.41 |     -21.19% |

Main finding:

```text
The strongest improvement was observed in soft tissue.
Bone improved but remained challenging.
Air/lung regions were not improved and remain a limitation.
```

---

## Masked-background SSIM

Whole-slice SSIM was found to be sensitive to background and outside-body regions.

Therefore, masked-background SSIM was also calculated by setting outside-mask pixels to air/background before SSIM calculation.

Final masked-background SSIM:

```text
Mean: 0.798
Std:  0.055
```

This showed better structural agreement within the relevant anatomical region than whole-slice SSIM.

---

## Ablation Study

Several experiments were performed during development:

| Experiment  | Description                                      |
| ----------- | ------------------------------------------------ |
| Day 10      | Combined loss, 30 training patients              |
| Day 12      | Region-weighted loss                             |
| Day 13      | Combined loss, 80 training patients, partial run |
| Final model | Combined loss, 80 training patients, 25 epochs   |
| MC dropout  | Exploratory uncertainty analysis                 |

The region-weighted loss improved bone and air/lung MAE slightly, but worsened overall MAE and soft-tissue performance. Therefore, the combined L1 + SSIM + gradient loss was selected for the final model.

---

## Monte Carlo Dropout Uncertainty

Monte Carlo dropout was implemented by keeping dropout active during inference and running multiple stochastic forward passes.

For each input slice:

```text
20 stochastic predictions
        ↓
mean prediction = MC mean sCT
standard deviation = uncertainty map
```

A random-100 validation analysis was performed using 100 randomly selected slices across 20 validation patients.

Random-100 MC dropout results:

| Metric                        |  Value |
| ----------------------------- | -----: |
| MC sCT MAE [HU]               |  97.22 |
| Raw CBCT MAE [HU]             | 193.74 |
| Mean uncertainty [HU]         |  41.81 |
| Error-uncertainty correlation |  0.147 |

The MC dropout mean prediction improved over raw CBCT, but did not outperform deterministic inference.

Final interpretation:

```text
Deterministic inference is used for final sCT generation.
MC dropout is reported as exploratory uncertainty analysis.
```

The uncertainty maps showed a weak positive relationship with prediction error but were not sufficiently calibrated for clinical reliability.

---

## Key Findings

* A 2.5D U-Net++ model can substantially improve CBCT HU accuracy.
* The final model reduced overall MAE from 216.97 HU to 88.84 HU.
* Soft tissue showed the strongest improvement.
* Bone improved but remained a difficult region.
* Air/lung regions remained challenging.
* Masked-background SSIM better represented structural quality than whole-slice SSIM.
* MC dropout uncertainty was technically implemented but showed weak calibration.
* A strong single-fold model was used due to Colab/T4 resource limitations.

---

## Repository Structure

```text
CBCT_to_sCT_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── metrics.py
│   ├── uncertainty.py
│   └── __init__.py
│
├── splits/
│   ├── fold_0.json
│   ├── fold_1.json
│   └── fold_2.json
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_and_crop_pad.ipynb
│   ├── 03_2p5d_dataset_testing.ipynb
│   ├── 04_training_pipeline_sanity_check.ipynb
│   ├── 05_metrics_and_evaluation_setup.ipynb
│   ├── 06_final_model_evaluation.ipynb
│   ├── 07_mc_dropout_uncertainty.ipynb
│   └── 08_final_results_and_figures.ipynb
│
└── results/
    ├── final_tables/
    └── final_project_figures/
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch with CUDA is not already installed, install it separately according to your CUDA version.

Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

Example final Colab training command:

```bash
python -m src.train \
    --fold 0 \
    --epochs 25 \
    --batch_size 2 \
    --max_train_patients 80 \
    --max_val_patients 20 \
    --loss_type combined \
    --l1_weight 1.0 \
    --ssim_weight 0.2 \
    --gradient_weight 0.05 \
    --checkpoint_dir checkpoints/final_single_fold \
    --log_dir results/metrics
```

Resume training:

```bash
python -m src.train \
    --fold 0 \
    --epochs 25 \
    --batch_size 2 \
    --max_train_patients 80 \
    --max_val_patients 20 \
    --loss_type combined \
    --resume_checkpoint checkpoints/final_single_fold/last_fold0.pth \
    --resume_log \
    --checkpoint_dir checkpoints/final_single_fold \
    --log_dir results/metrics
```

## Limitations

This project was not submitted to the official SynthRAD2025 challenge and was not trained using full competition-level resources.

Main limitations:

* Final training was limited by Google Colab/T4 GPU resources.
* Full-dataset training was not performed in the final Colab version.
* Full 3-fold or 5-fold ensembling was not performed.
* The current dataset class loads selected patient volumes into RAM, limiting scaling.
* Air/lung regions showed worse MAE after sCT generation.
* MC dropout uncertainty showed only weak correlation with prediction error.
* Dose calculation validation was not included.

---

## Future Work

Future improvements may include:

* Training on larger patient subsets using RTX 4090 or cluster resources.
* Implementing lazy loading or preprocessed slice caching to reduce RAM usage.
* Improving low-density air/lung region preservation.
* Testing alternative encoders such as EfficientNet or ConvNeXt.
* Using deep ensembles for better uncertainty estimation.
* Performing dose calculation validation for radiotherapy relevance.
* Comparing against diffusion-based CBCT-to-sCT approaches.

---

## Final Note

This repository represents a practical, resource-aware implementation of CBCT-to-synthetic CT generation for adaptive radiotherapy. The focus is on building a clean and interpretable medical imaging AI pipeline, including preprocessing, 2.5D model training, HU-based evaluation, region-wise analysis, and exploratory uncertainty estimation.

```
```
