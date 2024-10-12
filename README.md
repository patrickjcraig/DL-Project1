# DL-Project1
## Description
Convolutional neural networks(CNNs) and multi-layer perceptrons are foundational models for baselines in deep learning. They are useful for processing images, automatically determining relevant feature values, and testing novel hyperparameter selection, system blocks, and backpropagation alternatives.

The purpose of this repository is to test, on the Kuzushiji-MNIST dataset, the effects of different hyperparameter selection on the classification accuracy of these models.

## Model Architectures
### MLP

### CNN

## Hyperparameter Selection

### MLP

| Hyperparameter       | Value |
|----------------------|-------|
| Learning Rate        | 0.01  |
| Batch Size           | 64    |
| Number of Epochs     | 50    |
| Dropout Rate         | 0.5   |
| Optimizer            | Adam  |
| Weight Initialization| Xavier|

### CNN

| Hyperparameter       | Value |
|----------------------|-------|
| Learning Rate        | 0.001 |
| Batch Size           | 128   |
| Number of Epochs     | 100   |
| Dropout Rate         | 0.25  |
| Optimizer            | SGD   |
| Weight Initialization| He    |


## Classification Metrics
| Model                  | F1-score | Precision | Recall | Accuracy |
|------------------------|----------|-----------|--------|----------|
| CNN                    | 0.95     | 0.96      | 0.94   | 0.95     |
| Multi-layer Perceptron | 0.92     | 0.93      | 0.91   | 0.92     |

## Learning Rate
![Learning Rates](path/to/your/image.png)


## Relevant Links
- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [Deep Learning for Classical Japanese Literature](https://arxiv.org/pdf/1812.01718)
