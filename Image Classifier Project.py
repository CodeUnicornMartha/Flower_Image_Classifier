#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torchvision.models as models
#https://pytorch.org/docs/master/torchvision/models.html
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
#https://stackoverflow.com/questions/43441673/trying-to-load-a-custom-dataset-in-pytorch
#https://www.programcreek.com/python/example/104834/torchvision.transforms.Resize
#transforming the train data
data_train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])                                     
#trainsforming the test data
data_test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])   

#transforming the validation data
data_valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])   



# TODO: Load the datasets with ImageFolder
#loading the train/test/valid datasets
#load data and transform the data - apply on data
train_image_datasets = datasets.ImageFolder(train_dir,  transform=data_train_transforms)
test_image_datasets = datasets.ImageFolder(test_dir, transform=data_test_transforms)
valid_image_datasets = datasets.ImageFolder(valid_dir, transform=data_valid_transforms)



# TODO: Using the image datasets and the trainforms, define the dataloaders
#define the train/test/valid dataloader
train_dataloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=32,shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=32, shuffle = True)
valid_dataloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32,shuffle = True)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(json.dumps(cat_to_name, indent=4))


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[5]:


# TODO: Build and train your network
#train on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
model = models.vgg16(pretrained=True)


#https://www.programcreek.com/python/example/107699/torch.nn.Linear
#https://stackoverflow.com/questions/52268048/pytorch-saving-and-loading-a-vgg16-with-knowledge-transfer

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
#https://www.quora.com/What-is-the-VGG-neural-network
#https://www.cs.toronto.edu/~frossard/post/vgg16/
#https://www.programcreek.com/python/example/108009/torchvision.models.vgg16

#linear transformation from 7*7*512 to 4096 nodes - vgg transformed image 1/4*4096 -> 1024
#linear transformation from 4096 nodes to 102 categories
classifier = nn.Sequential(nn.Linear(7*7*512,4096),
                      nn.ReLU(),
                      nn.Dropout(p= 0.2),
                      nn.Linear(4096, 1024),
                      nn.ReLU(),
                      nn.Dropout(p= 0.2),
                      nn.Linear(1024, 102),
                      nn.LogSoftmax(dim=1))

#https://www.quora.com/What-is-the-VGG-neural-network
#replace the actual classification of the vgg16 architecture with my model
model.classifier = classifier

#defining the criterion loss function - difference from classified image to truth
criterion = nn.NLLLoss()
#defining the optimizer
#lr - how well the models adapts to new trained data
#forward/backward propagation according to Adam - adapt weights/gradient descent
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)


model.to(device);


# In[6]:


epochs = 20
steps = 0
running_loss = 0
print_every = 5

# load input
for epoch in range(epochs):
    for inputs, labels in train_dataloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Set Gradient to 0 and then Gradient Descent can be calculated
        optimizer.zero_grad()
        
        # Forward Propagation -> put input into model
        logps = model.forward(inputs)
        
        # calculate the loss = difference between truth and classified images
        loss = criterion(logps, labels)
        
        # Backward Propagation -> calculate the gradient descent...
        loss.backward()
        
        # ...adapt the weights
        optimizer.step()

        # store current loss
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            # Set the model to "evaluate" mode
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    #Loss of batch
                    batch_loss = criterion(logps, labels)
                    # store and aggregate batch loss in test loss
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    # top probabilities - top class
                    top_p, top_class = ps.topk(1, dim=1)
                    # top class equals the label truth
                    equals = top_class == labels.view(*top_class.shape)
                    # take the average of all correct classified images
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_dataloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(valid_dataloader):.3f}")
            running_loss = 0
            # Set the model to "train" mode
            model.train()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[7]:


accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        # https://knowledge.udacity.com/questions/28186
         # Calculate accuracy
        ps = torch.exp(logps)
        #top classified images
        top_p, top_class = ps.topk(1, dim=1)
        # check if top class is equals the truth
        equals = top_class == labels.view(*top_class.shape)
        #calculate the average
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()    

print(f"Accuracy of the Network on the test images: {accuracy/len(test_dataloader):.3f}")


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[8]:


# TODO: Save the checkpoint
# https://www.quora.com/What-is-the-VGG-neural-network
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
model.class_to_idx = train_image_datasets.class_to_idx
checkpoint = { 'classifier': model.classifier,
               'model_architecture':'vgg16',
               'state_dict': model.state_dict(),
               'optimizer_state': optimizer.state_dict(),
               'class_to_idx': model.class_to_idx,
                'epochs': epochs}
torch.save(checkpoint, 'checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[9]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
# https://knowledge.udacity.com/questions/28186
# https://knowledge.udacity.com/questions/32434
def load_checkpoint(filepath):
    import torch
    from torchvision import datasets, transforms
    import torchvision.models as models
    import numpy as np    
    #load checkpoint
    checkpoint = torch.load(filepath)
    
    if checkpoint['model_architecture'] == 'vgg16':
        model = models.vgg16(pretrained = True)
        
    elif model_architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
        
    elif model_architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
     
    elif model_architecture == 'densenet121':
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[10]:


def process_image(image):
    #https://mlpipes.com/pytorch-quick-start-classifying-an-image/
    #https://knowledge.udacity.com/questions/33724
    #https://knowledge.udacity.com/questions/29696
    #https://knowledge.udacity.com/questions/33724
    #https://mlpipes.com/pytorch-quick-start-classifying-an-image/
    #https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    #https://knowledge.udacity.com/questions/32474
    
    # define transforms
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]) 
    
    # preprocess the image
    pre_image = preprocess(image)
    
    np_image = np.array(pre_image, dtype = float)
    #https://discuss.pytorch.org/t/resizing-any-simple-direct-way/10316/5

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image/255
    np_image = (np_image - mean)/std
    #https://discuss.pytorch.org/t/axes-dont-match-array/37351/5
    #https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array
    np_image = np_image.transpose((2,0,1))

    # https://knowledge.udacity.com/questions/29696

    return np_image


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[11]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[12]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
    #https://knowledge.udacity.com/questions/29696
    #https://stackoverflow.com/questions/11727598/pil-image-open-working-for-some-images-but-not-others
    #https://mlpipes.com/pytorch-quick-start-classifying-an-image/
    
    from PIL import Image
    
    # open image
    image = Image.open(image_path)
    
    #https://knowledge.udacity.com/questions/33724
    # define transforms 
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
     
    # preprocess the image
    tensor_image = preprocess(image)

    model.to(device)
    
    # https://knowledge.udacity.com/questions/29696
    #https://mlpipes.com/pytorch-quick-start-classifying-an-image/
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device) 

    #get the labels from the model
    predicted_labels = model.forward(tensor_image)

    #calculate probability
    ps = torch.exp(predicted_labels)
 
    top_prob, top_class = torch.topk(ps, topk)

    #https://pynative.com/python-range-function/
    #https://knowledge.udacity.com/questions/29696
    #https://stackoverflow.com/questions/52074153/cannot-convert-list-to-array-valueerror-only-one-element-tensors-can-be-conver
    #https://stackoverflow.com/questions/1614236/in-python-how-do-i-convert-all-of-the-items-in-a-list-to-floats
    top_probs = [float(p) for p in top_prob[0]]

    #https://knowledge.udacity.com/questions/29696
    #https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    
    #https://knowledge.udacity.com/questions/29696
    #https://stackoverflow.com/questions/52074153/cannot-convert-list-to-array-valueerror-only-one-element-tensors-can-be-conver
    top_classes = [inv_map[int(i)] for i in top_class[0]]
    
    return top_probs, top_classes

# https://www.aiworkbox.com/lessons/convert-a-numpy-array-to-a-pytorch-tensor


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[13]:


# Sources: 
# https://github.com/ahmadabdolsaheb/ImageClassifier/blob/master/predict.py
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

import seaborn as sb
from PIL import Image
import json
import matplotlib.pyplot as plt


#load json
with open('cat_to_name.json', 'rb') as f:
    cat_to_name = json.load(f)

image_path = 'flowers/test/1/image_06752.jpg'
image = Image.open(image_path)

# TODO: Display an image along with the top 5 classes

# Get the probabilities and indices from passing the image through the model
probabilities, classes = predict(image_path, load_checkpoint('checkpoint.pth'), topk=5)

# Show Image -> imshow(image)
imshow(process_image(image))

# https://knowledge.udacity.com/questions/28639 
# plot bar charts with results
# receive all the names of the classes
names = [cat_to_name[i] for i in classes]

#https://plot.ly/matplotlib/bar-charts/
#https://plot.ly/matplotlib/bar-charts/#basic-bar-chart
#https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.barh.html
#https://stackoverflow.com/questions/34076177/matplotlib-horizontal-bar-chart-barh-is-upside-down
plt.figure()
#https://plot.ly/matplotlib/bar-charts/
#https://stackoverflow.com/questions/34076177/matplotlib-horizontal-bar-chart-barh-is-upside-down
#https://knowledge.udacity.com/questions/28639
plt.barh(np.arange(len(names)), probabilities, tick_label = names ,align='center', color = sb.color_palette()[0])
plt.ylabel('Classes')
plt.xlabel('Probabilities')
plt.title('Scores by classes and probabilities')
plt.yticks(np.arange(len(names)))
plt.gca().invert_yaxis() 
plt.show()

#https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
# https://knowledge.udacity.com/questions/28639 


# Sources used for Workspace 1 and Workspace 2 parts:
# https://github.com/ahmadabdolsaheb/ImageClassifier/blob/master/predict.py
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html
# https://pillow.readthedocs.io/en/latest/reference/Image.html
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://www.programcreek.com/python/example/108009/torchvision.models.vgg16
# https://www.quora.com/What-is-the-VGG-neural-network
# https://www.cs.toronto.edu/~frossard/post/vgg16/
# https://www.cs.toronto.edu/~frossard/post/vgg16/#architecture
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# https://pytorch.org/docs/0.3.0/torchvision/transforms.html
# https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# https://github.com/ahmadabdolsaheb/ImageClassifier/blob/master/predict.py
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# https://stackoverflow.com/questions/10873824/how-to-convert-2d-float-numpy-array-to-2d-int-numpy-array
# https://knowledge.udacity.com/questions/33306
# https://plot.ly/matplotlib/bar-charts/
# https://stackoverflow.com/questions/34076177/matplotlib-horizontal-bar-chart-barh-is-upside-down
# https://www.coursehero.com/file/p7ofeo15/ValueError-operands-could-not-be-broadcast-together-with-shapes-3-2-c-nparray7/
# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.barh.html
# https://knowledge.udacity.com/questions/33724
# https://knowledge.udacity.com/questions/32904
# https://knowledge.udacity.com/questions/32474
# https://knowledge.udacity.com/questions/31804
# https://mlpipes.com/pytorch-quick-start-classifying-an-image/
# https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
# https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
# https://stackoverflow.com/questions/53995708/axes-dont-match-array-in-pytorch
# https://stackoverflow.com/questions/48402009/given-input-size-128x1x1-calculated-output-size-128x0x0-output-size-is-t
# https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
# https://discuss.pytorch.org/t/resizing-any-simple-direct-way/10316/5
# https://stackoverflow.com/questions/48675114/pytorch-how-to-print-output-blob-size-of-each-layer-in-network
# https://stackoverflow.com/questions/51807040/typeerror-tensor-is-not-a-torch-image
# https://pynative.com/python-range-function/
# https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
# https://knowledge.udacity.com/questions/32474![image.png](attachment:image.png)
# https://discuss.pytorch.org/t/how-to-resize-a-tensor-or-image/17091/6![image.png](attachment:image.png)
# https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/
# https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
# https://stats.stackexchange.com/questions/63152/what-does-the-hidden-layer-in-a-neural-network-compute
# https://knowledge.udacity.com/questions/7854
# https://knowledge.udacity.com/questions/33306
# https://discuss.pytorch.org/t/runtimeerror-given-groups-1-weight-of-size-64-3-7-7-expected-input-3-1-224-224-to-have-3-channels-but-got-1-channels-instead/30153/5
# https://plot.ly/matplotlib/bar-charts/#basic-bar-chart![image.png](attachment:image.png)
# https://github.com/kg5528/flower_classifier/blob/master/predict.py![image.png](attachment:image.png)
