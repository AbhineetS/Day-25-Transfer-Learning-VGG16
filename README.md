# Day 25 — Transfer Learning with VGG16

This project demonstrates transfer learning using the VGG16 model (pre-trained on ImageNet) to classify images into two custom classes. Instead of training a CNN from scratch, we reuse VGG16 as a feature extractor and fine-tune the final block for better performance.

---

## 1. Project Overview

**Goal:**  
Build a high-accuracy 2-class image classifier using transfer learning.

**Key Idea:**  
Use VGG16 to extract image features and add a small custom classification head on top.  
This drastically reduces training time and improves results even with small datasets.

---

## 2. Dataset Structure

Your dataset must follow this format:

```
data/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
```

Each folder should contain images belonging to that class.  
The script automatically reads class names from these directories.

---

## 3. Model Architecture

**Base Model:** VGG16 (without the top layers)  
- Pretrained on ImageNet  
- All layers frozen in the first phase  
- Only used for feature extraction  

**Custom Classification Head:**
- GlobalAveragePooling2D  
- Dropout(0.3)  
- Dense(128, ReLU)  
- Dense(num_classes, Softmax)

**Fine-Tuning:**
- Unfreezes only the final block (`block5`)  
- Uses a low learning rate to prevent overfitting  

---

## 4. Training Workflow

### Phase 1: Train the custom head
- VGG16 remains frozen  
- Fast training  
- Optimizer: Adam (lr=1e-3)

### Phase 2: Fine-tune the last block
- Slightly adapts VGG16 to your dataset  
- Stable and effective  
- Optimizer: Adam (lr=1e-5)

---

## 5. How to Run

Make sure the `data/` directory exists, then run:

```
python3 train_transfer_vgg16.py
```

All generated outputs will be saved in the project folder.

---

## 6. Output Files

**Model File:**  
```
vgg16_transfer.keras
```

**Training Plot:**  
```
training_history_transfer.png
```

These files help you evaluate performance and reuse the trained model later.

---

## 7. Tools & Libraries

- TensorFlow / Keras  
- VGG16 Pretrained Weights  
- Matplotlib (for plotting training metrics)

---

## 8. Summary

This project shows how transfer learning can achieve strong results even with small datasets. VGG16 provides powerful feature extraction, and fine-tuning allows adaptation to your specific classification task. This technique is widely used in industry for fast, high-accuracy vision models.