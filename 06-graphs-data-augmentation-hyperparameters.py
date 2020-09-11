###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# Figures of the choosing of data augmentation techniques and hyperparameters 
####################################

# Import the libraries to do graphs
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the style of the seaborn graphs
sns.set_style("whitegrid")

#-----------------------------------------
# Plot of the learning rate validation
print('Making VGG19 learning rate graph')

# Route where the learning rate 
val_results_path = './results_validation_lr'

# List the files of the validation
files = os.listdir(val_results_path)

# get the files starning with vgg19
vgg19_files = list(filter(lambda k: 'vgg19' in k, files))

# Get the files and genenrate a dataframe with the compiled data
counter = 1
for i, csv_file in enumerate(vgg19_files):

    # Read the csv
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)

    # Put good names for the graph
    val_result['model'] = csv_file.replace('.csv','').replace('0_01','LR=0.01').replace('0_005','LR=0.005').replace('0_001','LR=0.001').replace('_',' ').replace('vgg','VGG')

    if i== 0:
        val_result_full = val_result
    else:
        val_result_full = pd.concat([val_result_full,val_result])


# Melt to generate a single graph with the training and validation accuracies
val_result_full_melt=pd.melt(val_result_full[['epoch','train_accuracy','val_accuracy','model']],id_vars= ['epoch','model'],value_vars=['train_accuracy','val_accuracy'])

# Rename variables for the graph
val_result_full_melt = val_result_full_melt.rename(columns={'variable': 'accuracy'},)
val_result_full_melt['accuracy'] = val_result_full_melt['accuracy'].str.replace('train_accuracy','Training').str.replace('val_accuracy','Validation')

# Make graph and save as pdf
ax_lr = sns.lineplot(x="epoch", y="value",style="accuracy", hue="model",data=val_result_full_melt).set(ylabel='Accuracy (%)',ylim=(0, 105))
plt.savefig('./document_latex/vgg19_lr.pdf')

# -----------------------------------------
# Graphs of the data augmentation techniques

print('Making VGG11 data augmentation graph')
# List the files of the validation
val_results_path = './results_validation'
files = os.listdir(val_results_path)

# Get the files that start with vgg11
vgg11_files = list(filter(lambda k: 'vgg11' in k, files))


counter = 1

# Read the csv datasets and generate a dataframe with all the data
for i, csv_file in enumerate(vgg11_files):

    # Read the csv files
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)
    val_result['model'] = csv_file.replace('.csv','').replace('_',' ').replace('vgg','VGG')

    if i== 0:
        val_result_full = val_result
    else:
        val_result_full = pd.concat([val_result_full,val_result])


# Order the colors
hue_order=['VGG11 NOAUG','VGG11 RH','VGG11 RC','VGG11 CJ','VGG11 CJ RH','VGG11 CJ RC','VGG11 RC RH','VGG11 CJ RC RH']

# Make a grid plot with training and validation accuracy and loss
fig, ax =plt.subplots(1,3)

#Go graph by braph
sns.lineplot(x="epoch", y="train_accuracy", hue="model",data=val_result_full,ax=ax[0],hue_order=hue_order).set(ylabel='Training Accuracy (%)',ylim=(0, 105))
ax[0].get_legend().remove()
sns.lineplot(x="epoch", y="val_accuracy", hue="model",data=val_result_full,ax=ax[1],hue_order=hue_order).set(ylabel='Validation Accuracy (%)',title="VGG11",ylim=(0, 105))
ax[1].get_legend().remove()
sns.lineplot(x="epoch", y="loss", hue="model",data=val_result_full,ax=ax[2],hue_order=hue_order).set(ylabel='Loss')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 6)

# Save the graph as pdf
fig.savefig('./document_latex/vgg11_aug.pdf',bbox_inches='tight')  



# Now the same but for the vgg19 data
print('Making VGG19 data augmentation graph')

# List the files of the validation
files = os.listdir(val_results_path)


vgg19_files = list(filter(lambda k: 'vgg19' in k, files))

vgg19_files

counter = 1

# Read the csv files and generate a dataset
for i, csv_file in enumerate(vgg19_files):

    # Read the csv files
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)
    val_result['model'] = csv_file.replace('.csv','').replace('_',' ').replace('vgg','VGG')

    if i== 0:
        val_result_full = val_result
    else:
        val_result_full = pd.concat([val_result_full,val_result])


# Make grid with the training and validation accuracy and the loss in each epoch
fig, ax =plt.subplots(1,3)

sns.lineplot(x="epoch", y="train_accuracy", hue="model",data=val_result_full,ax=ax[0],hue_order=['VGG19 NOAUG','VGG19 RH','VGG19 RC','VGG19 CJ','VGG19 CJ RH','VGG19 CJ RC','VGG19 RC RH','VGG19 CJ RC RH']).set(ylabel='Training Accuracy (%)',ylim=(0, 105))
ax[0].get_legend().remove()
sns.lineplot(x="epoch", y="val_accuracy", hue="model",data=val_result_full,ax=ax[1],hue_order=['VGG19 NOAUG','VGG19 RH','VGG19 RC','VGG19 CJ','VGG19 CJ RH','VGG19 CJ RC','VGG19 RC RH','VGG19 CJ RC RH']).set(ylabel='Validation Accuracy (%)',title="VGG19",ylim=(0, 105))
ax[1].get_legend().remove()
sns.lineplot(x="epoch", y="loss", hue="model",data=val_result_full,ax=ax[2],hue_order=['VGG19 NOAUG','VGG19 RH','VGG19 RC','VGG19 CJ','VGG19 CJ RH','VGG19 CJ RC','VGG19 RC RH','VGG19 CJ RC RH']).set(ylabel='Loss')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 6)

# Save the figure as pdf
fig.savefig('./document_latex/vgg19_aug.pdf',bbox_inches='tight')  










