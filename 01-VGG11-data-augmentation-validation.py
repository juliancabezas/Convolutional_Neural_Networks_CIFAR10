###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# VGG11 implementation on the CIFAR-10 dataset, using different data augmentation techniques
####################################

# Importing the pytorch models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler

#----------------------------------------
# PARAMETERS
# Folder to put the outputs of the trainings
results_folder = "./results_validation/"

# Number of epochs and learning rate
n_epoch = 5
learning_rate = 0.01

# Data augmendation techniques to apply
COLOR_JITTER = True
RANDOM_CROP = True
RANDOM_HORIZONTAL_FLIP = True

# Name to put in the csv output files
modelname = "vgg11"

#----------------------------------------

# Depending on the chosen data augmentation methods I will build the name of the model
if COLOR_JITTER:
    modelname = modelname + '_CJ'
if RANDOM_CROP :
    modelname = modelname + '_RC'
if RANDOM_HORIZONTAL_FLIP:
    modelname = modelname + '_RH'


# Generate the set of transformations
transformations_selected = []

# Include the corresponding transformation if the parameter is set to TRUE
if COLOR_JITTER:
    transformations_selected.append(transforms.ColorJitter(0.5,0.5,0.5))
if RANDOM_CROP :
    transformations_selected.append(transforms.RandomCrop(32, padding = 4))
if RANDOM_HORIZONTAL_FLIP:
    transformations_selected.append(transforms.RandomHorizontalFlip())

# The noramlization to let the data be between -1 and +1 is always applied
transformations_selected.append(transforms.ToTensor())
transformations_selected.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

# Set the training data augmentations
transform_train = transforms.Compose(transformations_selected)

# In the case of the validation and test datasets, the only data augmentation that is applied is the normalization
transform_val_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Download the training  and validation data  (We are download the train data two times, we will the separate it later)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

valset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_val_test)


# I will get  a separation between the training and the validation data

# Get the indexes of the 50000 data in the train dataset
indices = list(range(50000))
train_len = 40000

# Randomization of the indexes
np.random.seed(41)
np.random.shuffle(indices)

# Subset the data and use th samplers to get the train and validation data
train_index = indices[:train_len]
val_index = indices[train_len:]
train_sampler = SubsetRandomSampler(train_index)
val_sampler = SubsetRandomSampler(val_index)


# Download the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_val_test)


# Data loaders, in the train and validation data we are using the samplers to subset the data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,sampler=train_sampler,
                                          num_workers=4, generator=torch.Generator().manual_seed(58))
valloader = torch.utils.data.DataLoader(valset, batch_size=10,sampler=val_sampler,
                                         num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         num_workers=4)


# Set cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Class of the VGG11 neural network, it has to inherit from nn.Module
class VGG11_CIFAR10(nn.Module):
    def __init__(self):
        # Call super constructor of the class
        super(VGG11_CIFAR10, self).__init__()

        # Convolution blocks
        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
        #self.conv1_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)

        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        #self.conv2_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)

        self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        #self.conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        #self.conv3_4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)

        self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        #self.conv4_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        #self.conv4_4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

        self.conv5_1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        #self.conv5_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        #self.conv5_4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

        # Pooling layer, to be used between each group of convolution layers
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully connected layers, the last one outputs the class
        self.fc1 = nn.Linear(in_features = 512 * 1 * 1, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512,  out_features = 512)
        self.fc3 = nn.Linear(in_features = 512,  out_features = 10)

        # Use Kaiming initialization for the convolutional layers, just as in the torchvision implemntation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            # Initialize the linear layers with normal distributuoon close to zero
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Forward method, we use a ReLU in each step
    def forward(self, x):
        
        # Stacks of convolutionnal layers and max pooling

        x = F.relu(self.conv1_1(x)) # out: n, 64, 32,  32
        #x = F.relu(self.conv1_2(x)) # out: n, 64, 32,  32
        x = self.pool(x) # out: n, 64, 16, 16

        x = F.relu(self.conv2_1(x)) # out: n, 128, 16,  16
        #x = F.relu(self.conv2_2(x)) # out: n, 128, 16,  16
        x = self.pool(x) # out: n, 128, 8, 8

        x = F.relu(self.conv3_1(x)) # out: n, 256, 8,  8
        x = F.relu(self.conv3_2(x)) # out: n, 256, 8,  8
        #x = F.relu(self.conv3_3(x)) # out: n, 256, 8,  8
        #x = F.relu(self.conv3_4(x)) # out: n, 256, 8,  8
        x = self.pool(x) # out: n, 256, 4, 4

        x = F.relu(self.conv4_1(x)) # out: n, 512, 4,  4
        x = F.relu(self.conv4_2(x)) # out: n, 512, 4,  4
        #x = F.relu(self.conv4_3(x)) # out: n, 512, 4,  4
        #x = F.relu(self.conv4_4(x)) # out: n, 512, 4,  4
        x = self.pool(x) # out: n, 512, 2, 2

        x = F.relu(self.conv5_1(x)) # out: n, 512, 2, 2
        x = F.relu(self.conv5_2(x)) # out: n, 512, 2, 2
        #x = F.relu(self.conv5_3(x)) # out: n, 512, 2, 2
        #x = F.relu(self.conv5_4(x)) # out: n, 512, 2, 2
        x = self.pool(x) # out: n, 512, 1, 1

        # Flatten to get a vector of length 515
        x = x.view(-1, 512 * 1 * 1)

        # Fully connected layers, adapted to match the resolution and number of classes of CIFAR-10
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.5,training=True)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,0.5,training=True)
        x = self.fc3(x)

        return x

# Call the constructor, set seed for reproducibility
torch.manual_seed(45)
torch.cuda.manual_seed(45)
net = VGG11_CIFAR10().to(device)

# We will use Cross Entropy Loss and backpropagation to train the NN, using the defined learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# Empty list to store the results
epoch_full = []
acc_full = []
acc_full_val = []
loss_full = []

# Training of the neural network
for epoch in range(1,n_epoch+1):  # loop over the dataset multiple times

    print("Epoch:",epoch)

    # Initializing the trainign variables
    running_loss_full = 0.0
    correct_total_train = 0
    correct_total_val = 0
    nbatch = 0
    nsamples_train = 0
    nsamples_val = 0

    # Go though all the training data
    for i, data in enumerate(trainloader, 0):
        print(i, end='\r')

        # Get the inputs (images) and labels (target variable)
        inputs, labels = data
        # Put them in the graphics card
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Set the parameters to zero
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Calculate the  gradients (backward)
        loss = criterion(outputs, labels)
        loss.backward()
        # Backpropagation
        optimizer.step()

        # Measure the loss
        running_loss_full += loss.item() 
        nbatch = nbatch + 1
        

    # Get the accuracy in the training data
    with torch.no_grad():
        for data in trainloader:
            images_train, labels_train = data
            images_train = images_train.to(device)
            labels_train = labels_train.to(device)
            # Generate prediction without gradient updat and calculate the correct matches
            outputs_train = net(images_train)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train = (predicted_train == labels_train).float().sum()
            correct_total_train = correct_total_train + correct_train.item()
            nsamples_train = nsamples_train + len(labels_train)  

    train_accuracy = 100.0 * correct_total_train / nsamples_train
    print("Train Accuracy:", train_accuracy)

    # Get the accuracy in the validation data
    with torch.no_grad():
        for data in valloader:
            images_val, labels_val = data
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = net(images_val)
            # Generate prediction without gradient updat and calculate the correct matches
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val = (predicted_val == labels_val).float().sum()
            correct_total_val = correct_total_val + correct_val.item()
            nsamples_val = nsamples_val + len(labels_val)  

    val_accuracy = 100.0 * correct_total_val / nsamples_val
    print("Validation Accuracy:", val_accuracy)

    # Store the accuracies and loss in the lists
    acc_full.append(train_accuracy)
    acc_full_val.append(val_accuracy)
    loss_full.append(running_loss_full / nbatch)
    epoch_full.append(epoch)

    print("Loss:", running_loss_full / nbatch)


print('Finished Training')

# Create pandas dataset and store the results in a csv
dic = {'epoch':epoch_full,'train_accuracy':acc_full,'val_accuracy':acc_full_val,'loss':loss_full}
df_grid_search = pd.DataFrame(dic)
df_grid_search.to_csv(results_folder + modelname + '.csv')


