
# Assignment 2, Deep Learning Fundamentals, 2020

Julian Cabezas Pena. 
Student ID: a1785086

Testing data augmentation techniques and VGG convolutional neural networks for multiclass image classification on the CIFAR-10 dataset, using Pytorch

## Environment

This repo was tested under a Linux 64 bit OS, using Python 3.7.7, PyTorch 1.6.0 and TorchVision 0.7.0

It was also tested in Google Colab (https://colab.research.google.com/) with GPU enabled


## How to run this repo

In order to use this repo:

1. Clone or download this repo

```bash
git clone https://github.com/juliancabezas/Convolutional_Neural_Networks_CIFAR10.git
```

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)
3. Create a environment using the vgg_environment.yml file included in this repo, using the following command (inside conda or bash)

```bash
conda env create -f vgg_environment.yml --name vgg_environment
```

4. Activate the conda environment

```bash
conda activate vgg_environment
```

5. Run each specific file in yout IDE of preference, (I recommend [VS Code](https://code.visualstudio.com/) with the Python extension), using the root folder of the directory as working directory to make the relative paths work.

It is also possible to run the codes openning a terminal in the project directory
```bash
python <name_of_the_py_file>
```
* Alternatevely, you can use Google Colab (https://colab.research.google.com/) with GPU enabled

Run the codes in order:
- 00-Figures-images-cifar10.py: Creates the figures with samples of the CIFAR10 dataset and of the applied data augmentation methods (Optional)
- 01-VGG11-data-augmentation-validation.py: Run the VGG11 architecture using different data augmentation techniques adjusting the parameters
- 02-VGG19-data-augmentation-validation.py: Run the VGG19 architecture using different data augmentation techniques adjusting the parameters
- 03-VGG19-learning-rate-validation.py: Run the VGG19 architecture using different learning rates (hyperparameter adjusting)
- 04-VGG19-final-model-train.py: Train the final model of the VGG19 architecture, using Random Horizontal Flip and Random Crop with learning rate 0.01
- 05-VGG19-final-model-test.py: Use the final model on the test dataset, and get the accuracies
- 06-Graphs-data-augmentation-hyperparameters.py: Graphs of data augmentation validation and learning rate adjusting (Optional)





