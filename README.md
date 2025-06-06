## Signal Quality Classifier

A robust neural network-based classifier to predict signal quality (Signal Strength) using a multi-layer perceptron (MLP) implemented with TensorFlow/Keras. This project demonstrates advanced machine learning techniques, including data preprocessing, model regularization, and performance optimization, to achieve stable and accurate predictions.

## 📖 Overview
The Signal Quality Classifier project focuses on predicting signal strength (ranging from 3 to 8) using a dataset of 11 input parameters. The model is a multi-layer perceptron (MLP) built with TensorFlow/Keras, incorporating dropout for regularization and class weights to handle imbalanced data. The project showcases best practices in data preprocessing, model training, and evaluation, achieving a validation accuracy of ~70–75% with reduced overfitting.
This repository is ideal for machine learning practitioners interested in classification tasks, neural network optimization, and handling imbalanced datasets.

## ✨ Features

Neural Network Model: A 3-layer MLP with dropout (0.2) for regularization.
Data Preprocessing: Standardization of features and label encoding for the target variable.
Class Imbalance Handling: Utilizes class weights to address imbalanced signal strength distribution.
Performance Visualization: Training and validation loss/accuracy plots for model evaluation.
Environment: Built and tested in Google Colab with GPU acceleration.


## 📊 Dataset
The dataset (NN Project Data - Signal.csv) contains 1599 samples with 11 input features (e.g., Parameter 1 to Parameter 11) and a target variable Signal_Strength (integer values from 3 to 8). The distribution of signal strength is imbalanced, with the majority of samples having values of 5 and 6, as shown below:


## 🛠️ Setup Instructions
#Prerequisites

# Python 3.7+
Libraries: tensorflow, pandas, scikit-learn, matplotlib
Optional: Google Colab with GPU support for faster training

# Installation

Clone the repository:git clone https://github.com/your-username/signal-quality-classifier.git
cd signal-quality-classifier


Create a virtual environment and activate it:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:pip install -r requirements.txt

Note: If requirements.txt is not present, install the packages manually:pip install tensorflow pandas scikit-learn matplotlib



# Dataset

Place the NN Project Data - Signal.csv file in the root directory of the project.
Alternatively, upload the dataset to Google Colab if running the notebook there.


## 🚀 Usage

Open the Jupyter notebook:
jupyter notebook signal_quality_classifier.ipynb

Or, if using Google Colab, upload signal_quality_classifier.ipynb and the dataset to your Colab environment.

Run the notebook cells sequentially to:

# Load and preprocess the data.
Train the neural network model.
Evaluate and visualize performance.


The model performance plots will be saved as outputs/improved_model_performance.png.



## 📈 Results
The improved model demonstrates:

Validation Accuracy: ~70–75%, a significant improvement over the baseline.
Reduced Overfitting: Smaller gap between training and validation metrics due to dropout.
Stable Convergence: Smoother loss and accuracy curves over 50 epochs.

Performance Plots

## Insights

Dropout Regularization: Adding dropout (0.2) reduced overfitting, as seen in the smaller train-validation gap.
Class Weights: Improved handling of imbalanced classes, boosting performance on minority classes.
Convergence: Smoother training curves indicate better stability compared to the baseline model.



## Built with 💻 and ☕ by Kapil Tanwar
