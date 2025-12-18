# 🧠 Material Stream Identification System
**Oject Classification using CNN Features and Classical Machine Learning**

## 📌 Overview
The **Material Stream Identification System** is an end-to-end machine learning pipeline for classifying waste materials into six categories:

**Glass, Paper, Cardboard, Plastic, Metal, Trash**

The system integrates **deep learning feature extraction (ResNet50)** with **classical machine learning classifiers (Support Vector Machine and k-Nearest Neighbors)** to achieve high classification accuracy while maintaining computational efficiency.  
It supports **offline evaluation, model comparison, and real-time webcam deployment**.

This project was developed as part of a **Machine Learning course project**, focusing on the performance trade-offs between feature representation techniques and classification algorithms in image-based material identification.

---

## 👥 Team Members
| Name | ID |
|------|----|
| Ali Mohsen | 20221106 |
| Sherif Mahmoud | 20221080 |
| Mostafa Mohamed | 20221153 |
| Youssef Waleed | 20221206 |
| Noureldin Yacer | 20220462 |

---

## 🎯 Project Objectives
- Compare **hand-crafted features** with **deep learning features**
- Evaluate the performance of **SVM vs k-NN** classifiers
- Analyze **accuracy, training time, and computational trade-offs**
- Deploy a **real-time waste material classification system**
- Support **unknown object detection** using confidence-based rejection

---

## 🧠 Feature Extraction

### 1️⃣ Deep Learning Features (Primary)
- **Model**: Pre-trained **ResNet50**
- **Weights**: ImageNet
- **Output**: 2048-dimensional feature vector
- **Method**: Network truncated before the final classification layer
- **Advantages**:
  - High-level semantic representations
  - Strong class separability
  - Excellent performance with classical classifiers

### 2️⃣ Hand-Crafted Features (Legacy / Comparison)
- Local Binary Patterns (LBP)
- Gray-Level Co-occurrence Matrix (GLCM)
- Histogram of Oriented Gradients (HOG)
- Color Histograms (HSV)
- Color Moments  

**Total feature dimensionality**: 2303

---

## 🤖 Classification Models

### 🔹 Support Vector Machine (SVM)
- Kernel: RBF
- Parameters:
  - `C = 10`
  - `gamma = 'scale'`
  - `probability = True`
- Strengths:
  - High accuracy
  - Robust margin maximization
- Trade-off:
  - Higher training time

### 🔹 k-Nearest Neighbors (k-NN)
- Neighbors: `k = 5`
- Distance Metric: **Cosine**
- Voting: Distance-weighted
- Strengths:
  - Very fast training
  - Performs well with CNN features
- Trade-off:
  - Slower inference time

---

## 📊 Experimental Results

### CNN Feature Performance
| Model | Test Accuracy | Validation Accuracy | Training Time |
|------|---------------|---------------------|---------------|
| **SVM** | **98.54%** | 97.32% | 109.16 s |
| **k-NN** | 97.44% | 96.15% | 5.12 s |

### Key Observations
- CNN features outperform hand-crafted features by **7–12%**
- SVM achieves the highest classification accuracy
- k-NN becomes competitive in well-structured feature spaces
- CNN features significantly reduce SVM training time

---

## 🧱 Project Structure
├── augmentation_pipeline.py      # Dataset balancing & augmentation (flip, rotation, brightness, noise)
├── cnn_feature_extraction.py     # Feature extraction using pre-trained ResNet50 (ImageNet)
├── feature_extraction.py         # Hand-crafted feature extraction (legacy/alternative)
├── train_classifiers.py          # End-to-end training & evaluation pipeline
├── svm_classifier.py             # SVM model configuration (RBF, C=10, gamma=scale)
├── knn_classifier.py             # KNN model configuration (k=5, cosine distance)
├── realtime_classifier.py        # Real-time webcam-based classification
├── test.py                       # Evaluation script for hidden/test dataset
├── best_model.pkl                # Saved best model, scaler, and configuration
└── README.md                     # Project documentation

