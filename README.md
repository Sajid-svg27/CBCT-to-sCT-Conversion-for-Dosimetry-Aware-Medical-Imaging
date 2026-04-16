
# Deep Learning-Based CBCT-to-sCT Conversion for Dosimetry-Aware Medical Imaging

## 📌 Overview

This project implements a 3D deep learning pipeline for converting Cone-Beam CT (CBCT) images into synthetic CT (sCT) images using the SynthRAD2025 dataset.  

Unlike standard image-to-image translation tasks, this work emphasizes **dosimetric relevance**, evaluating not only image similarity but also **Hounsfield Unit (HU) accuracy, electron density consistency, and depth-dose behavior**.

---

## 🎯 Objectives

- Generate high-quality synthetic CT (sCT) from CBCT scans  
- Achieve strong structural similarity (SSIM, PSNR)  
- Evaluate **quantitative HU accuracy**  
- Assess **dosimetric consistency** using PDD-like analysis  
- Investigate limitations of purely deep learning-based approaches  

---

## 📂 Dataset

- **Dataset:** SynthRAD2025 (Task 2)
- Modalities:
  - CBCT (`cbct.mha`)
  - Planning CT (`ct.mha`)
  - Body mask (`mask.mha`)
- Regions used: Head & Neck (HN), Abdomen (AB) *(configurable)*  
- Each patient volume is 3D with aligned voxel spacing and geometry  

---

## 🧠 Methodology

### 🔹 Model Architecture
- 3D U-Net
- Residual learning (predicts correction over CBCT)
- Instance normalization
- Encoder-decoder with skip connections

---

### 🔹 Training Strategy
- Patch-based training (48×48×48)
- Random spatial sampling within body mask
- Mixed precision training (AMP)
- AdamW optimizer
- Cosine learning rate scheduling

---

### 🔹 Loss Function

Combination of:

- L1 Loss (intensity accuracy)
- SSIM Loss (structural similarity)
- Edge Loss (gradient consistency)

```math
Loss = 2.0 * L1 + 0.2 * SSIM + 0.3 * Edge
