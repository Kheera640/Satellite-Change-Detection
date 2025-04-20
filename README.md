# 🌍 Satellite Image Change Detection using Deep Learning

This project implements a **ResNet50-based Siamese U-Net model** to detect structural or land-use changes between two satellite images taken at different times. It uses the **LEVIR-CD+** dataset and is built using **TensorFlow/Keras**.

---

## 📌 Features

- ⚙️ Siamese architecture with shared ResNet50 encoder
- 🧠 Decoder using Conv2DTranspose layers for upsampling
- 🛰️ Works on bi-temporal satellite imagery
- 🔄 Feature subtraction and activation to highlight changes
- 🎯 Output: Binary mask of detected changes
- 🚀 Supports data augmentation, preprocessing, and tf.data pipelines

---

## 📁 Dataset

- Dataset: [LEVIR-CD+](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd)
- Format:
