# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Reference:
# - https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html

"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        self.data = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
        for file in data_files:
            batch = unpickle(file)
            self.data.append(batch[b'data'])    # b used to denote byte string
            self.labels.extend(batch[b'labels'])
        
        self.data = np.concatenate(self.data).reshape(-1,3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.labels = np.array(self.labels) # turn into array for 
    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.data)
        

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            idx:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        img, label  = self.data[idx], self.labels[idx]
        return img, label


def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    if "tr" in mode:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # TODO: what should be normalized here?

        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])
    return transform

def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    return CIFAR10(data_files, transform)

"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
     # Extract necessary parameters from the loader_params dictionary
    batch_size = loader_params.get("batch_size", 1)  # Default batch size is set to 1
    shuffle = loader_params.get("shuffle", False)    # Default shuffle is set to False
    
    # Additional DataLoader parameters can be passed using **loader_params ？
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
    


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.model = resnet18(pretrained=False)
        checkpoint = torch.load('resnet18.pt')
        self.model.load_state_dict(checkpoint)
        # 2. Freeze convolutional backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # 3. Initialize linear layer(s)
        num_features = self.model.fc.in_features
        num_classes = 8
        self.model.fc = nn.Linear(num_features, num_classes)
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
         # Get the features from the backbone
        # features = self.backbone(x)
        # # Flatten the features
        # features = torch.flatten(features, start_dim=1)
        # # Get the classification logits
        # logits = self.classifier(features)
        # # Normalize the logits
        # logits = F.log_softmax(logits, dim=1)
        x = x.reshape(-1, 3, 32, 32)
        x = torch.tensor(x, dtype=torch.float)
        return self.model(x)
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(params = model_params, **hparams)
    else:
        optimizer = torch.optim.SGD(params = model_params, **hparams)
    return optimizer 
 

"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################

    for img, label in train_dataloader:
        label = F.one_hot(label.long(), num_classes=8)  # Convert label to index tensor
        label_pred = model(img)

        loss = loss_fn(label_pred, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    # test_loss = something
    # print("Test loss:", test_loss)
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for img, label in test_dataloader:
            total+= label.size(0)
            label_pred  = model(img)
            pred_class = label_pred.argmax(dim=1)
            correct += (pred_class == label).sum().item()  

    test_loss /= len(test_dataloader)
    test_acc = correct / total
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    return test_acc




# A helper debug 
def test_file():
    data_files = [
        "cifar10_batches/data_batch_1",
        "cifar10_batches/data_batch_2",
        "cifar10_batches/data_batch_3",
        "cifar10_batches/data_batch_4",
        "cifar10_batches/data_batch_5"
    ]
    data = []
    labels = []
    for file in data_files:
        batch = unpickle(file)
        data.append(batch[b'data'])    # b used to denote byte string
        labels.extend(batch[b'labels'])    
    # print(data)
    print(len(labels))

    data = np.concatenate(data).reshape(-1,3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    labels = np.array(labels) # turn into array for 
    print(data.shape)
    print(labels.shape)

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    # Set up data files and parameters
    data_files = [
        "cifar10_batches/data_batch_1",
        "cifar10_batches/data_batch_2",
        "cifar10_batches/data_batch_3",
        # "cifar10_batches/data_batch_4",
        # "cifar10_batches/data_batch_5"
    ]
    test_batch = ["cifar10_batches/test_batch",]
    SGD_hparams = {
        'lr': 0.002,

        'momentum': 0.9
    }
    Adam_hparams = {
        'lr': 0.001
    }
    optim_type = 'SGD'
    loader_params = {'batch_size': 64, 'shuffle': True}

    train_dataset = build_dataset(data_files, transform= get_preprocess_transform("train"))
    test_dataset = build_dataset(test_batch, transform= get_preprocess_transform("test"))


    train_dataloader = build_dataloader(train_dataset,loader_params)
    test_dataloader = build_dataloader(test_dataset,loader_params)
    print("finished building data set")

    fine_model = build_model()
    print("finished building model")

    Cross_Etp_loss = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(optim_type, fine_model.parameters(),SGD_hparams)
    train(train_dataloader, fine_model, Cross_Etp_loss, optimizer)
    print("finished training")
    test(test_dataloader,fine_model)
    print("finished testing")

    return fine_model

    
