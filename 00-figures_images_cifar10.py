###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# Figures of CIFAR-10 dataset and data augmentation technigues
####################################

# Importing the necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Figure of the training data in CIFAR-10

# No data transformation
transform_train = transforms.Compose(
    [transforms.ToTensor()])

# Download the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)


# Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          num_workers=4, generator=torch.Generator().manual_seed(58))


classes = ['airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# get 100 random images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# I will pick 10 images, one for each label
images_sub = []
labels_sub = []

# I am going to retrieve one image per class
for labelnum_collect in range(0,10):
    count = 0
    for labelnum in labels:
        if labelnum.item() == labelnum_collect:
            images_sub.append(torch.index_select(images, 0, torch.tensor([count])))
            labels_sub.append(torch.index_select(labels, 0, torch.tensor([count])))
            print('Finding label:', labelnum_collect)
            break
        count +=1

# Convert to torch tensor
images_sub = torch.cat(images_sub, dim=0)

# Make a grid plot with 5 rows and 1 column for the 10 images
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
count=0

for row in ax:
    for col in row:

        # Get the 'first' dimension of the tensor and then eliminate it with squezze
        im = torch.index_select(images_sub, 0, torch.tensor([count]))
        im = im.squeeze(0)

        # I need to transpose to get the RGB
        col.imshow(im.numpy().transpose(1,2,0))

        # Put the title (class name)
        col.set_title(classes[labels_sub[count]],fontsize=35)
        count += 1


# Save the image
plt.savefig('./document_latex/samples_images.pdf',bbox_inches='tight')




# -----------------------------------------------

# Figure of the data transformations

# I will only use the first 5 images
images_sub = images_sub[0:5]
labels_sub = labels_sub[0:5]
classes_sub = classes[0:5]

# Get the images with the horizontal flip
transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])

# Download the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)


# Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          num_workers=4, generator=torch.Generator().manual_seed(58))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

images_sub_RH = []
labels_sub_RH = []

# I am going to retrieve the same 5 images
for labelnum_collect in range(0,5):
    count = 0
    for labelnum in labels:
        if labelnum.item() == labelnum_collect:
            images_sub_RH.append(torch.index_select(images, 0, torch.tensor([count])))
            labels_sub_RH.append(torch.index_select(labels, 0, torch.tensor([count])))
            print('Finding label:', labelnum_collect)
            break
        count +=1

images_sub_RH = torch.cat(images_sub_RH, dim=0)
images_sub_RH.size()

# Now the images with the Random crop
transform_train = transforms.Compose(
    [transforms.RandomCrop(32, 4),
     transforms.ToTensor()])

# Download the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

# Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          num_workers=4, generator=torch.Generator().manual_seed(58))

# get some random training images 
dataiter = iter(trainloader)
images, labels = dataiter.next()

images_sub_RC = []
labels_sub_RC = []

# I am going to retrieve the same 5 images
for labelnum_collect in range(0,5):
    count = 0
    for labelnum in labels:
        if labelnum.item() == labelnum_collect:
            images_sub_RC.append(torch.index_select(images, 0, torch.tensor([count])))
            labels_sub_RC.append(torch.index_select(labels, 0, torch.tensor([count])))
            print('Finding label:', labelnum_collect)
            break
        count +=1

images_sub_RC = torch.cat(images_sub_RC, dim=0)


# Images with the Color Jitter transformation
transform_train = transforms.Compose(
    [transforms.ColorJitter(0.5,0.5,0.5),
     transforms.ToTensor()])

# Download the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)


# Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          num_workers=4, generator=torch.Generator().manual_seed(58))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

images_sub_CJ = []
labels_sub_CJ = []

# I am going to retrieve the same 5 images
for labelnum_collect in range(0,5):
    count = 0
    for labelnum in labels:
        if labelnum.item() == labelnum_collect:
            images_sub_CJ.append(torch.index_select(images, 0, torch.tensor([count])))
            labels_sub_CJ.append(torch.index_select(labels, 0, torch.tensor([count])))
            print('Finding label:', labelnum_collect)
            break
        count +=1

images_sub_CJ = torch.cat(images_sub_CJ, dim=0)






# Now I will make a figure with 4 rows, the first with the original images and then the 3 transformations
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(16, 16))

count=0
rowcount = 0

# Go through the rows and columns of the subplot
for row in ax:
    for col in row:

        # Depending on the row take the correspondind data or title
        if rowcount == 0:
            images_sub_row = images_sub
            classes_sub_row = ['','','Original Image','','']
        elif rowcount == 1:
            images_sub_row = images_sub_RH
            classes_sub_row = ['','','Random Horizontal Flip (RH)','','']
        elif rowcount == 2:
            images_sub_row = images_sub_RC
            classes_sub_row = ['','','Random Crop (RC)','','']
        elif rowcount == 3:
            images_sub_row = images_sub_CJ
            classes_sub_row = ['','','Color Jitter (CJ)','','']

        # Get the channel of the torch vector
        im = torch.index_select(images_sub_row, 0, torch.tensor([count]))
        im = im.squeeze(0)

        col.imshow(im.numpy().transpose(1,2,0))

        # Set the title according to the data transformation, it will only print the thris (center)
        col.set_title(classes_sub_row[labels_sub[count]],fontsize=40)
        count += 1
    

    rowcount += 1
    count=0

# Save the plot
plt.savefig('./document_latex/samples_images_transformation.pdf',bbox_inches='tight')

print("Figures ready")








