# Plant Disease Classification using a 5-Layer CNN

## Overview

This project implements a Convolutional Neural Network (CNN) for plant disease classification using the PlantVillage dataset.

The objective is to classify plant leaf images into disease categories by learning visual features such as discoloration, spots, and lesion patterns.

---

## Dataset

Dataset: PlantVillage

The dataset contains labeled images of healthy and diseased plant leaves.

### Selected Classes

- Tomato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Potato Healthy
- Potato Early Blight

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Resize images to 128 × 128
- Random Horizontal Flip
- Random Rotation
- Normalization
- Tensor Conversion

Dataset Split:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

---

## CNN Architecture

The model consists of 5 learnable layers:

1. Conv Layer (3 → 16)
2. Conv Layer (16 → 32)
3. Conv Layer (32 → 64)
4. Fully Connected Layer
5. Output Layer

Architecture Flow:

Input Image
→ Conv1
→ MaxPool
→ Conv2
→ MaxPool
→ Conv3
→ Flatten
→ FC1
→ FC2 (Output)

---

## Performance Metrics

| Metric | Value |
|----------|----------|
| Test Accuracy | 50% |
| Precision | 0.25 |
| Recall | 0.50 |
| F1 Score | 0.33 |

---

## Deployment Analysis

| Parameter | Value |
|------------|----------|
| Model Parameters | 6,446,498 |
| Model Size | 24 MB |
| Training Epochs | 10 |

### Deployment Feasibility

- Laptop/Desktop: Supported
- Cloud Deployment: Supported
- Raspberry Pi: Possible
- FPGA Deployment: Possible after optimization

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

## Repository Structure

```text
Plant-Disease-Classifier/
│
├── Plant_Disease_Classification.ipynb
├── plantcnn.pth
├── requirements.txt
├── README.md
```

---

## Future Improvements

- Train for additional epochs
- Add Batch Normalization
- Apply Quantization
- Use Transfer Learning (ResNet/MobileNet)
- Deploy as a Streamlit Web App

---

## Author

Vaishnavi Mudda

BITS Pilani Hyderabad Campus

M.Sc. Physics + ENI
