# Project: Comparison of Optimization Algorithms

This project aims to compare different optimization algorithms used in image classification for a simple machine learning model. Specifically, it implements Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Adam to optimize a model that classifies generated images into two distinct classes. The main focus is to understand and compare how these algorithms perform on simple datasets and observe their convergence behaviors.

## Overview

The assignment is split into two main parts:

### Part A: Image Classification

### Part B: Visualization of the Optimization Process

The entire project is written in C, with the visualization part being implemented in Python for effective data analysis.

### Part A: Image Classification

Dataset

The dataset consists of two classes of synthetically generated images:

Class A (Dogs)

Class B (Cats)

Each image is:

Grayscale, with pixel values ranging from 0 to 1 (normalized from the original range of 0 to 255).

Represented as an N x N vector, where .

There are 100 samples per class for a total of 200 images.

The dataset is split into 80% training and 20% testing sets.

## Model

The model takes the input image and produces a classification output using the formula:

### Output:

Where:

w is the weight vector to be learned (including a bias term).

x is the input vector (flattened image).

The weight vector, w, has a dimension of (N^2 + 1) (where the extra +1 accounts for the bias).

## Optimization Algorithms

The project compares three different optimization methods to train the model:

Gradient Descent (GD)

Stochastic Gradient Descent (SGD)

## Adam Optimization

The comparison involves running each algorithm with 5 different initializations of weights and comparing their performances using:

Time taken for convergence.

Number of updates required to minimize the loss.

Training and test loss during epochs.

## Metrics for Comparison

Graphs are plotted to analyze the algorithms using the following metrics:

Time vs. Loss

Epoch vs. Loss

## Bonus Task

Additionally, the classification task was extended to handle a 4-class dataset for a more complex comparison.

Part B: Visualization of the Optimization Process

Visualization with T-SNE

In this part, the weight trajectories during training are visualized to observe how each optimization method converges to a solution.

The following process is used:

Record Weights: Record the weight vectors after each update during the training process for each of the 5 initial values of weights.

Dimensionality Reduction: Apply T-SNE to reduce the recorded weight vectors to 2 dimensions.

Visualization: Plot the 5 trajectories in a single graph to compare how each initialization and optimization algorithm evolves over time.

The visualization code is implemented in Python using Matplotlib and Scikit-Learn's T-SNE.

## Files and Structure

The project consists of the following files and their respective purposes:

### C Files

main.c: Entry point to run the training and testing of the model.

image_generator.c: Generates synthetic images for both classes (dogs and cats).

image_data.c: Handles dataset creation, normalization, and splitting.

model.c: Defines the model, training methods, and the optimization algorithms.

train.c: Coordinates the training and testing process for each optimization method.

## Headers

image_data.h: Declarations related to image data operations.

image_generator.h: Declarations for generating synthetic images.

model.h: Declarations related to the model, optimization functions, and training routines.

Python Files

visualize.py: Handles the visualization of the optimization process using T-SNE to analyze the trajectories of weights during training.

CMake File

CMakeLists.txt: Configuration file to compile the entire project with CMake.

## Getting Started

Prerequisites

C Compiler: GCC or Clang for compiling the C files.

Python: Python 3.x with Matplotlib and Scikit-Learn installed.

## How to Compile and Run

Clone the Repository:

git clone <repository_url>
cd <repository_directory>

Build the Project with CMake:

mkdir build
cd build
cmake ..
make

## Run the Program:

./run_training

Visualize Optimization Trajectories:

python visualize.py

## Results

The project compares the performance of Gradient Descent, SGD, and Adam for different initializations of weights. The results are plotted to help visualize:

How quickly each algorithm converges.

The stability of each optimization method.

Differences in how each method explores the weight space (visualized using T-SNE).

## License

This project is licensed under the MIT License.

## Acknowledgments

Scikit-Learn for providing the T-SNE implementation used in visualization.

Matplotlib for data visualization.

Author

Daniel Alves - Initial Work

