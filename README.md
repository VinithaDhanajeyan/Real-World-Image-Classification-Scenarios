# Real-World-Image-Classification-Scenarios

This project demonstrates an end-to-end image classification pipeline using a Convolutional Neural Network (CNN) built with **PyTorch**. A custom dataset organized into class folders is used to train and evaluate the model.

---
## Colab link :
https://colab.research.google.com/drive/1APwxrVS3ThlsaQe8h70dC_nxXagG42AV?usp=sharing

## 📁 Project Structure
```
- *Source*: [Kaggle Dataset](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)
- The dataset contains three categories stored in subfolders:

- animals/
├── cats/
├── dogs/
└── pandas/
```

---

## 🧠 Model Architecture

The CNN consists of:

- **Conv2D + ReLU + MaxPooling**
- **Flatten layer**
- **Fully Connected (FC) layers**
- **Softmax for multi-class classification**

This structure allows efficient feature extraction and classification of input images.

---

## 🧪 Dataset

- **Type**: Custom image dataset
- **Structure**: Folder-based, where each folder represents one class.
- **Sample size**: ~1000 images per class
- **Image size**: Resized to 128x128 during preprocessing

---

## 🚀 Getting Started

### 🔧 Installation

Make sure you have the following installed:
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- scikit-learn (for confusion matrix)

Install required libraries:

```bash
pip install torch torchvision matplotlib scikit-learn
```

# 📊 Performance
The model is trained over several epochs. The following metrics are tracked:

Training and Validation Accuracy

Training and Validation Loss

(Optional) Confusion Matrix

# 🔍 Example Graphs
Accuracy vs Epochs	Loss vs Epochs
<img width="1341" height="553" alt="image" src="https://github.com/user-attachments/assets/150b5106-8b0d-4f11-98ed-fcccd26c579a" />

# ✅ Evaluation
Final model accuracy: 97.2%

<img width="592" height="515" alt="image" src="https://github.com/user-attachments/assets/91dcbbf3-2f43-492d-bbae-a6164aa6c890" />


# 💡 Challenges Faced
Dataset imbalance between classes

Overfitting during early training

GPU memory limits

# Solutions:

Used DataLoader with shuffling

Early stopping & dropout (if implemented)

Batch size tuning
