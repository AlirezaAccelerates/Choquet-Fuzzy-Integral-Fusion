# Choquet Fuzzy Integral Fusion
This repository contains the implementation of a Choquet Fuzzy Integral Fusion Model. The Choquet Integral is a powerful aggregation operator used in multi-criteria decision making and image processing.

## Overview
The Choquet Fuzzy Integral Fusion Model is designed to combine multiple input features in a way that captures the interactions between them. This model can be particularly useful in scenarios where the importance of each feature and the synergy between them are not constant but vary depending on the context.

## Installation
To use this code, you need to have Python installed along with the necessary packages. You can install the required packages using the following command:
```bash
pip install numpy cvxopt
```

## Usage
Initialization
To instantiate the ChoquetIntegral class:
```python
from choquet_integral import ChoquetIntegral

chi = ChoquetIntegral()
```
## Training
To train the model, you need to provide the training samples and their corresponding labels. The training samples should be a NumPy array of size N x M (inputs x number of samples), and the labels should be a NumPy array of size 1 x M (label per sample).
```
train_samples = np.array(...)  # Replace with your training samples
train_labels = np.array(...)   # Replace with your training labels

chi.fit(train_samples, train_labels)
```

## Prediction
To make predictions with the trained model, provide the test sample:
```
test_sample = np.array(...)  # Replace with your test sample

prediction = chi.predict(test_sample)
print("Prediction:", prediction)
```

## Code Explanation
The main class in this implementation is ChoquetIntegral, which provides methods for training (fit), predicting (predict), and other internal functions necessary for building and using the Choquet Integral.

### Class and Methods
ChoquetIntegral: Initializes the Choquet Integral model.
fit(x1, l1): Trains the model using the provided training samples and labels.
predict(x2): Predicts the output for a given test sample.
produce_lattice(): Builds the lattice of fuzzy measure (FM) variables.
build_constraint_matrices(index_keys, fm_len): Constructs the constraint matrices needed for the quadratic program.
get_fm_class_img_coeff(Lattice, h, fm_len): Creates a fuzzy measure map with the name as the key and the index as the value.
get_keys_index(): Sets up a dictionary for referencing FM components.
Example
Here's a brief example demonstrating how to use the ChoquetIntegral class:
```
import numpy as np
from choquet_integral import ChoquetIntegral

# Instantiate the model
chi = ChoquetIntegral()

# Define training samples and labels
train_samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
train_labels = np.array([0.7, 0.8, 0.9])

# Train the model
chi.fit(train_samples, train_labels)

# Define a test sample
test_sample = np.array([0.2, 0.3])

# Make a prediction
prediction = chi.predict(test_sample)
print("Prediction:", prediction)
```

If you are interested, please cite:
```
@inproceedings{rafiei2022automated,
  title={Automated major depressive disorder classification using deep convolutional neural networks and Choquet fuzzy integral fusion},
  author={Rafiei, Alireza and Wang, Yu-Kai},
  booktitle={2022 IEEE Symposium Series on Computational Intelligence (SSCI)},
  pages={186--192},
  year={2022},
  organization={IEEE}
}
```

