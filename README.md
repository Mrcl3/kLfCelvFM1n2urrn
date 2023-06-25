# MonReader: Page Flipping Detection

This project focuses on predicting whether a page is being flipped using a single image. It aims to provide a solution for the MonReader mobile document digitization experience, specifically for detecting page flips in real-time.

## Project Overview

MonReader is a mobile app that offers fully automatic, high-speed, and high-quality document scanning. By leveraging artificial intelligence and computer vision technologies, MonReader's goal is to streamline the document digitization process for various users, including the blind, researchers, and anyone in need of bulk document scanning.

## Dataset

The dataset used in this project consists of video clips recorded from smartphones, capturing both page flipping and non-flipping actions. These videos have been labeled accordingly. The frames from these videos have been extracted, saved as individual images.


## Project Steps

1. Data Preparation:
   - Download the dataset using the provided link.
   - Extract the dataset files and inspect the structure.
   - Verify the correctness of the labels, ensuring the presence of "flipping" and "not flipping" categories.
   - Split the dataset into training and testing sets.

2. Model Training:
   - Preprocess the images to prepare them for training.
   - Select a deep learning framework, such as TensorFlow or PyTorch.
   - Design a convolutional neural network (CNN) model architecture.
   - Train the model using the labeled images and their corresponding labels.
   - Optimize the model's performance by tuning hyperparameters.

3. Model Evaluation:
   - Evaluate the trained model using the testing dataset.
   - Calculate the F1 score as the primary evaluation metric.
   - Analyze additional metrics such as accuracy, precision, and recall for further insights.
   - Iterate on the model and hyperparameters based on evaluation results.

4. Bonus: Predicting Flipping Sequences:
   - Extend the project to predict if a given sequence of images contains a flipping action.
   - Utilize recurrent neural networks (RNNs) or 3D convolutional neural networks (3D CNNs) to capture temporal dependencies in the image sequences.
   - Prepare a dataset of labeled image sequences for training and testing.
   - Adapt the model architecture to handle sequential data.
   - Evaluate the model's performance on a separate testing set of image sequences.

## Success Metrics

The model's performance will be evaluated based on the F1 score, which considers both precision and recall. The higher the F1 score, the better the model's ability to predict page flipping accurately.

## Dependencies

Specify the dependencies required to run the code in this project. For example:
- Python 3.7 or above
- TensorFlow 2.5 or above
- NumPy


