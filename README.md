# ML_Project
CIFAR-10 Image Classification Project

Overview

This project involves the classification of images from the CIFAR-10 dataset using various machine learning and deep learning techniques. The goal is to compare the performance of different models on the same dataset.

Dataset

Name: CIFAR-10
Description: A collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
Source: https://www.cs.toronto.edu/~kriz/cifar.html
Project Structure

ML_Proje-2.ipynb: Main Jupyter Notebook containing all the code.
dataset/cifar-10-batches-py/: Folder containing the CIFAR-10 dataset in binary format.
Methods Used

Data Preprocessing
Normalization of pixel values
Reshaping images for model input
Machine Learning Models
Random Forest Classifier
XGBoost Classifier
Multi-layer Perceptron (MLP)
KMeans Clustering (for analysis)
Deep Learning Model
Convolutional Neural Network (CNN) using TensorFlow/Keras
Evaluation Metrics

Accuracy
F1 Score
Confusion Matrix
Classification Report
Matthews Correlation Coefficient (MCC)
Silhouette Score (for clustering)
How to Run

Download the CIFAR-10 dataset and extract it into the dataset/cifar-10-batches-py/ directory.
Open ML_Proje-2.ipynb in Jupyter Notebook or JupyterLab.
Run all the cells in sequence to reproduce the results.
Dependencies

Python 3.x
TensorFlow
Keras
NumPy
Pandas
Scikit-learn
XGBoost
Matplotlib
Seaborn
Install dependencies using:

pip install -r requirements.txt
Results

The project demonstrates the performance differences between traditional ML models and CNNs on image data, highlighting the superior accuracy of CNNs for image classification tasks.