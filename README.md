# Handwritten Digit Recognition

A comparison of three neural network architectures — a single-layer Perceptron, a Multi-Layer ANN, and a CNN — trained on the MNIST handwritten digit dataset using TensorFlow/Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Richik06/Handwritten-Digit-Recognition/blob/main/HandwrittenDigitRecognition.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)

## Overview

This project explores how model complexity affects performance on a classic image classification task. Starting from the simplest possible model and working up to a convolutional network, it trains and evaluates each architecture on the same data and compares their accuracy, loss curves, and confusion matrices side by side.

## Results

| Model | Architecture | Test Accuracy |
|---|---|---|
| Perceptron | Single dense layer (softmax) | 90.85% |
| ANN | 2 hidden dense layers (ReLU) | 97.63% |
| CNN | 2 Conv2D + MaxPooling + Dropout | **99.09%** |

All three models were trained for 5 epochs with a batch size of 32. Full training/validation curves, a validation accuracy comparison plot, and a confusion matrix for the CNN are generated in the notebook.

## Dataset

This project uses the **MNIST dataset in CSV format** — 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels, flattened into 784 pixel columns plus one `label` column.

- **Training set:** 60,000 images
- **Test set:** 10,000 images

`mnist_test.csv` is already included in this repository. `mnist_train.csv` is not included because of its size — download both files from the links below and place them in the project root before running the notebook.

- 📦 Train set: [mnist_train.csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv)
- 📦 Test set: [mnist_test.csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv)
- Source: [MNIST in CSV — Kaggle (oddrationale)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## Tech Stack

- Python
- NumPy, Pandas
- Matplotlib, Seaborn (visualization)
- scikit-learn (preprocessing, metrics)
- TensorFlow / Keras (model building & training)

## Project Structure

```
Handwritten-Digit-Recognition/
├── HandwrittenDigitRecognition.ipynb   # Main notebook: preprocessing, training, evaluation
├── mnist_test.csv                      # Test data (included)
├── mnist_train.csv                     # Train data (download separately, see Dataset section)
└── README.md
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
```

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Richik06/Handwritten-Digit-Recognition.git
   cd Handwritten-Digit-Recognition
   ```
2. Download `mnist_train.csv` from the [dataset link](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) above and place it in the project root (`mnist_test.csv` is already there).
3. Open `HandwrittenDigitRecognition.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run all cells in order.

Alternatively, click the **Open in Colab** badge at the top of this README to run the notebook directly in your browser (you'll still need to upload the CSV files to the Colab session).

## What the Notebook Covers

1. Loading and exploring the MNIST CSV data
2. Preprocessing: normalization and reshaping for each model type
3. Building and training three models — Perceptron, ANN, CNN
4. Evaluating test accuracy for each model
5. Visualizing training/validation accuracy and loss curves
6. Comparing all three models on a single validation accuracy chart
7. Generating a confusion matrix for the best-performing model (CNN)
8. Side-by-side visual comparison of predictions across all models

## Possible Improvements

- Add data augmentation to improve CNN generalization
- Tune hyperparameters (learning rate, batch size, epochs) for each model
- Add early stopping and learning rate scheduling
- Save trained models and build a simple inference script or web demo

## Author

**Richik06** — [GitHub Profile](https://github.com/Richik06)
