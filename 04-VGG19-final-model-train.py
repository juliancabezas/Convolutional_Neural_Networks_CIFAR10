###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# VGG19 implementation on the CIFAR-10 dataset, using random horizontal flip and random crop
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

#---------------------------------------
# Folder to store the model object
model_folder = "./models/"

# Number of epochs
n_epoch = 100

# Learning rate, picked from the best validation  set accuracy
learning_rate = 0.01

# Model name (for the .pth file)
modelname = "vgg19_RC_RH_final"

#------------------------------

# Transforms the data to tensor, apply RH and RC transformations
transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# In the test data we only apply the normalization
transform_val_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Download the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

# Download the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_val_test)



# Data loader of the train and test dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          num_workers=4, generator=torch.Generator().manual_seed(58))

testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         num_workers=4)


classes = ('airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# functions to show an image

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images, nrow=10))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(10)))

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


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train with the dataset multiple times (n epoch)
for epoch in range(1,n_epoch+1):  

    print("Epoch:",epoch)

    # Initializing the training variables
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

        # Set the gradients to zero
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
            # Get the output and see how they match with the true labels
            outputs_train = net(images_train)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train = (predicted_train == labels_train).float().sum()
            correct_total_train = correct_total_train + correct_train.item()
            nsamples_train = nsamples_train + len(labels_train)  

    train_accuracy = 100.0 * correct_total_train / nsamples_train
    print("Train Accuracy:", train_accuracy)
    print("Loss:", running_loss_full / nbatch)


print('Finished Training')

# Specify a path
PATH = model_folder + modelname + '.pth'

# Save the model
torch.save(net.state_dict(), PATH)