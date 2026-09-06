# 🌸 Flower Classification AI

A deep learning project that classifies flower images into 5 different categories using a Convolutional Neural Network (CNN).

## 🌼 Flower Classes

The model can classify images into:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

## 🧠 Model

The project uses a CNN built using TensorFlow and Keras.

The model includes:

- Image resizing to 180 × 180
- Image normalization
- Data augmentation
- Convolutional layers
- Max pooling
- Dense layers
- Softmax output for 5 flower classes

## 📊 Model Performance

The model was trained for 10 epochs.

- Training Accuracy: ~75%
- Validation Accuracy: ~72.3%

The model was also evaluated using a confusion matrix and classification report.

## 🚀 Streamlit Application

A Streamlit web application allows users to upload a flower image and receive:

- Predicted flower class
- Prediction confidence
- Probability for each flower class
- Low-confidence warning

### Application Flow

```text
Upload Image
      ↓
Image Preprocessing
      ↓
CNN Model
      ↓
Class Probabilities
      ↓
Predicted Flower
      ↓
Confidence Score