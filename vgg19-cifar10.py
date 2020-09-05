# Importing the pytorch moduels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 


""" 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True) """


# Transforms the data to tensor
#transform = transforms.Compose(
#    [transforms.ToTensor()])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Download the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=1)

# Download the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=1)

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


class Net(nn.Module):
    def __init__(self):
        # Call super constructor of the class
        super(Net, self).__init__()

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
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features = 512 * 1 * 1, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512,  out_features = 512)
        self.fc3 = nn.Linear(in_features = 512,  out_features = 10)


        #self.classifier = nn.Sequential(
        #nn.Dropout(),
        #nn.Linear(512, 512),
        #nn.ReLU(True),
        #nn.Dropout(),
        #nn.Linear(512, 512),
        #nn.ReLU(True),
        #nn.Linear(512, 10),
        #)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        torch.nn.init.xavier_uniform_(m.weight)
        #        torch.nn.init.zeros_(m.bias)
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()



    def forward(self, x):

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


        #x = x.view(-1, 512 * 1 * 1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.5,training=True)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,0.5,training=True)
        x = self.fc3(x)

        #x = self.classifier(x)

        return x


net = Net().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

epoch_full = []
acc_full = []
loss_full = []

for epoch in range(200):  # loop over the dataset multiple times

    print("Epoch:",epoch+1)

    running_loss = 0.0
    running_loss_full = 0.0
    correct_total = 0
    nbatch = 0

    for i, data in enumerate(trainloader, 0):
        print(i, end='\r')
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # Calulate overall accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).float().sum()
        correct_total = correct_total + correct.item()

        running_loss_full += loss.item() 

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        nbatch = nbatch + 1
        nsamples = nsamples + len(labels)  
        

    train_accuracy = 100.0 * correct_total / nsamples
    print("Accuracy:", train_accuracy)

    acc_full.append(train_accuracy)
    loss_full.append(running_loss_full / nbatch)
    epoch_full.append(epoch+1)


print('Finished Training')

# Create pandas dataset and store it in a csv
dic = {'epoch':epoch_full,'accuracy':acc_full,'loss':loss_full}
df_grid_search = pd.DataFrame(dic)
df_grid_search.to_csv('./drive/My Drive/deeplearning/train_vgg19_v2.csv')

# Specify a path
PATH = "./models/vgg19_state_dict.pt"

# Save
#torch.save(net.state_dict(), PATH)

#net = Net().to(device)
net.load_state_dict(torch.load(PATH))
net.eval()


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

print("hola")


import platform
print(platform.python_version())

import torch
print(torch.__version__)