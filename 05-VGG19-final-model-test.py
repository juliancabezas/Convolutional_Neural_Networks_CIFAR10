###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# VGG19 implementation on the CIFAR-10 dataset, using random horizontal flip and random crop - Test
####################################


# Importing the necessary libraries
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

#------------------------------------
# Parameters
model_folder = "./models/"
modelname = "vgg19_RC_RH_final"

#----------------------------------

# We will only use the test data, so no transformation
transform_val_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


# Download the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_val_test)

# Set the test loader
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         num_workers=4)

# Get a tuple with the classes
classes = ('airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Set cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Class of the VGG11 neural network, it has to inherit from nn.Module
class VGG19_CIFAR10(nn.Module):
    def __init__(self):
        # Call super constructor of the class
        super(VGG19_CIFAR10, self).__init__()

        # Convolution blocks
        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)

        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)

        self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv3_4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)

        self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv4_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv4_4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

        self.conv5_1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv5_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv5_4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

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
        x = F.relu(self.conv1_2(x)) # out: n, 64, 32,  32
        x = self.pool(x) # out: n, 64, 16, 16

        x = F.relu(self.conv2_1(x)) # out: n, 128, 16,  16
        x = F.relu(self.conv2_2(x)) # out: n, 128, 16,  16
        x = self.pool(x) # out: n, 128, 8, 8

        x = F.relu(self.conv3_1(x)) # out: n, 256, 8,  8
        x = F.relu(self.conv3_2(x)) # out: n, 256, 8,  8
        x = F.relu(self.conv3_3(x)) # out: n, 256, 8,  8
        x = F.relu(self.conv3_4(x)) # out: n, 256, 8,  8
        x = self.pool(x) # out: n, 256, 4, 4

        x = F.relu(self.conv4_1(x)) # out: n, 512, 4,  4
        x = F.relu(self.conv4_2(x)) # out: n, 512, 4,  4
        x = F.relu(self.conv4_3(x)) # out: n, 512, 4,  4
        x = F.relu(self.conv4_4(x)) # out: n, 512, 4,  4
        x = self.pool(x) # out: n, 512, 2, 2

        x = F.relu(self.conv5_1(x)) # out: n, 512, 2, 2
        x = F.relu(self.conv5_2(x)) # out: n, 512, 2, 2
        x = F.relu(self.conv5_3(x)) # out: n, 512, 2, 2
        x = F.relu(self.conv5_4(x)) # out: n, 512, 2, 2
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

# Call the constructor
torch.manual_seed(45)
torch.cuda.manual_seed(45)
net = VGG19_CIFAR10().to(device)


# Specify the mpth of the trained model
PATH = model_folder + modelname + '.pth'

# Load model
net.load_state_dict(torch.load(PATH))
net.eval()

# Not we are going to print the accuracy for each of the models
# (Sorry this code is very similar to the workshop)
correct_total_test = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# Make sure of using no_grad to leave the weigths as they are
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        # Get the predicted values for the batch and see how many are correct
        _, predicted = torch.max(outputs, 1)
        correct_test = (predicted == labels).float().sum()
        correct_total_test = correct_total_test + correct_test.item()
        c = (predicted == labels).squeeze()
        # Get the correct items class by class
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print the per class results and the final results
for i in range(10):
    print("Accuracy of " + classes[i] + ": " + "{:.2%}".format(class_correct[i] / class_total[i]))

print("Overall test Accuracy: "+"{:.2%}".format(correct_total_test/10000))
