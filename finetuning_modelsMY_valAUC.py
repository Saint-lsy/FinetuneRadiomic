# -*- coding: UTF-8 -*-
"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim
#import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
#import local_models
#import matplotlib.pyplot as plt
import time
import os
import copy
import random
import glob
import numpy as np
from torch.backends import cudnn
from sklearn.metrics import roc_auc_score

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

writer = SummaryWriter('tensorboard/experiment1')
seed = 7
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) 
random.seed(seed)
np.random.seed(seed)
cudnn.benchmark = False          
cudnn.deterministic = True
# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "I:\\DL_Data\\CAMS_ZL\\Quarter_AreaRate050_image\\Group\\TRAIN_Twice\\TRAIN_1"
data_dir_2 = "I:\\DL_Data\\CAMS_ZL\\Quarter_AreaRate050_image\\Group\\TRAIN_Twice\\TRAIN_2"
data_dir_all = 'I:\\DL_Data\\CAMS_ZL\\Quarter_AreaRate050_image\\Group\\TRAIN'
exVal_dir = 'I:\\DL_Data\\CAMS_ZL\\Quarter_AreaRate050_image\\Group\\test301'
exVal_dir2 = 'I:\\DL_Data\\CAMS_ZL\\Quarter_AreaRate050_image\\Group\\testFK'



# data_dir = '/data/train_val_test/ln3_A/'
data_dir = '/data/data_lsy/tvt_64/ln3_A/'
typeln = 'ln3_A'
# data_dir_2 = '/data/train_val_test/ln3_A/'

exVal_dir = os.path.join(data_dir, 'test')
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet" #"squeezenet" 

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for 
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True


class MyDataset(Dataset):
    def __init__(self, file_path, transform = None, target_transform = None):
        """
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.datas = []
        self.labels = []
        img_paths= glob.glob(os.path.join(file_path, "*.npy")) 
        for path in img_paths:
            img, label= np.load(path, allow_pickle=True)   #通过index索引返回一个图像路径fn 与 标签label
            for i in range(3):
                self.data.append(img[:,:,i])
                self.labels.append(label)
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img, label= np.load(self.img_paths[index], allow_pickle=True)   #通过index索引返回一个图像路径fn 与 标签label

        label = int(label)
        if self.transform is not None:
            img = self.transform(img) 
        return img, label              #这就返回一个样本
    
    def __len__(self):
        return len(self.img_paths)          #返回长度，index就会自动的指导读取多少

class MyDataset(Dataset):
    def __init__(self, file_path, transform = None, target_transform = None):
        """
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.img_paths = []
        self.img_paths= glob.glob(os.path.join(file_path, "*.npy"))                
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img, label= np.load(self.img_paths[index], allow_pickle=True)   #通过index索引返回一个图像路径fn 与 标签label
        # if int(label)== 0:
        #     label = np.array([1,0])
        # else:
        #     label = np.array([0, 1])
        # label = np.array(label)
        # label = torch.FloatTensor(label)
        # img = torch.tensor(img)
        label = int(label)
        if self.transform is not None:
            img = self.transform(img) 
        return img, label              #这就返回一个样本
    
    def __len__(self):
        return len(self.img_paths)          #返回长度，index就会自动的指导读取多少


class FocalLossV1(nn.Module):
 
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
 
        self.celoss = torch.nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''
 
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        ce_loss=(-(label * torch.log(logits)) - (
                    (1 - label) * torch.log(1 - logits)))
        # ce_loss=(-(label * torch.log(torch.softmax(logits, dim=1))) - (
        #             (1 - label) * torch.log(1 - torch.softmax(logits, dim=1))))
        pt = torch.where(label == 1, logits, 1 - logits)
        # ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss



def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        starttime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            TP, TN, FP, FN =0, 0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)  # preds == labels.data, 返回的torch元素为0或1,不是bool类型
                #NPpreds = preds.data.cpu().numpy()   #GPU tensor不能直接转为numpy数组，必须先转到CPU tensor
                TP += torch.sum((preds.data == labels.data) & (labels.data == 1))
                TN += torch.sum((preds.data == labels.data) & (labels.data == 0))
                FP += torch.sum((preds.data != labels.data) & (labels.data == 0))
                FN += torch.sum((preds.data != labels.data) & (labels.data == 1))


            # if phase == 'val':
            #     print(pred_score)
            #     epoch_auc = roc_auc_score(labels.data.cpu().numpy(),pred_score)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_sens = TP.double() / (TP + FN)
            epoch_spec = TN.double() / (TN + FP)
            #测试代码
            # acc2 = (TP.double() + TN.double()) / (TP+FN+TN+FP)
            # print(acc2,epoch_acc,'相等就对了')
            print('{} Loss: {:.4f} Acc: {:.4f} sens: {:.4f} spec {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_sens, epoch_spec))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                sens = epoch_sens
                spec = epoch_spec
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                endtime = time.time()
                print ('Total time of this epoch:%f' % (endtime-starttime))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} with sens {:4f} and spec {:4f}'.format(best_acc, sens, spec))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

        


#######################  External Validation  

def ExValidation(model, exvalDataloaders, criterion):

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    TP, TN, FP, FN =0, 0, 0, 0
    for inputs, labels in exvalDataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            
            pred_score = outputs[:,1].cpu().detach().numpy()            

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)  # preds == labels.data, 返回的torch元素为0或1,不是bool类型
        #NPpreds = preds.data.cpu().numpy()   #GPU tensor不能直接转为numpy数组，必须先转到CPU tensor
        #下面这种只在二分类适用的以后要进行普适化更改，if 类别数 == 2：
        TP += torch.sum((preds.data == labels.data) & (labels.data == 1))
        TN += torch.sum((preds.data == labels.data) & (labels.data == 0))
        FP += torch.sum((preds.data != labels.data) & (labels.data == 0))
        FN += torch.sum((preds.data != labels.data) & (labels.data == 1))

    loss = running_loss / len(exvalDataloaders.dataset)
    acc = running_corrects.double() / len(exvalDataloaders.dataset)
    sens = TP.double() / (TP + FN)
    spec = TN.double() / (TN + FP)
    print('External validation \n Loss: {:.4f} Acc: {:.4f} sens: {:.4f} spec {:.4f} '.format(loss, acc, sens, spec))

    return acc,sens,spec







def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, num_classes),
        #     # nn.Softmax(dim=1),
        #     # nn.Sigmoid()
        # )
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=1),
            nn.Sigmoid()
        )
        input_size = 224


#    elif model_name == "densenet32":
#        from local_models import densenet
#        model_ft = densenet.densenet32(pretrained=use_pretrained)
#        set_parameter_requires_grad(model_ft, feature_extract)
#        num_ftrs = model_ft.classifier.in_features
#        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
#        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        #对预训练网络的最后一层进行替换
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...") 
        exit()
    
    return model_ft, input_size 

# Initialize the model for this run
#model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
print(model_ft) 


######################################################################
# Load Data
# ---------
# 
# Now that we know what the input size must be, we can initialize the data
# transforms, image datasets, and the dataloaders. Notice, the models were
# pretrained with the hard-coded normalization values, as described
# `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
# 

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),#不转换为PIL会报错
        transforms.Resize(input_size),        
        # transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.681, 0.681, 0.681], [0.10, 0.10, 0.10])   #imagenet
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),#不转换为PIL会报错
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.68, 0.68, 0.68], [0.10, 0.10, 0.10])
    ]),
}

exVal_transforms = transforms.Compose([
        transforms.ToPILImage(),#不转换为PIL会报错
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.68, 0.68, 0.68], [0.10, 0.10, 0.10])
    ])
    
print("Initializing Datasets and Dataloaders...")



# Create training and validation datasets
image_datasets = {x: MyDataset(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
train_data = MyDataset(os.path.join(data_dir,'train'))

train_weights = []
for i in range(len(train_data)):
    _, label = train_data[i]  
    if label == 0:
        train_weights.append(1/383)
    else:
        train_weights.append(1/168)     
train_weights = torch.FloatTensor(train_weights)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

val_data = MyDataset(os.path.join(data_dir,'val'))
val_weights = [] 
for i in range(len(val_data)):
    _, label = val_data[i]  
    if label == 0:
        val_weights.append(1/82)
    else:
        val_weights.append(1/36)      
val_weights = torch.FloatTensor(val_weights)
val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_data))


dataloaders_dict = {'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                            batch_size=batch_size, 
                                            shuffle=False, 
                                            num_workers=0,
                                            sampler = train_sampler
                                            ),
                    'val':torch.utils.data.DataLoader(image_datasets['val'], 
                                            batch_size=len(val_data), 
                                            shuffle=True, 
                                            num_workers=0,
                                            # sampler = val_sampler
                                            )
                }

# Detect if we have a GPU available
if not torch.cuda.is_available() == True:
    raise IOError('The cuda is not available!')
device = torch.device("cuda:0") #if torch.cuda.is_available() else "cpu"
#device = torch.device("cpu")


######################################################################
# Create the Optimizer
# --------------------
# 
# Now that the model structure is correct, the final step for finetuning
# and feature extracting is to create an optimizer that only updates the
# desired parameters. Recall that after loading the pretrained model, but
# before reshaping, if ``feature_extract=True`` we manually set all of the
# parameter’s ``.requires_grad`` attributes to False. Then the
# reinitialized layer’s parameters have ``.requires_grad=True`` by
# default. So now we know that *all parameters that have
# .requires_grad=True should be optimized.* Next, we make a list of such
# parameters and input this list to the SGD algorithm constructor.
# 
# To verify this, check out the printed parameters to learn. When
# finetuning, this list should be long and include all of the model
# parameters. However, when feature extracting this list should be short
# and only include the weights and biases of the reshaped layers.
# 

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=0.005, momentum=0.9)
optimizer_ft = optim.Adam(params_to_update, lr=0.005,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)


######################################################################
# Run Training and Validation Step
# --------------------------------
# 
# Finally, the last step is to setup the loss for the model, then run the
# training and validation function for the set number of epochs. Notice,
# depending on the number of epochs this step may take a while on a CPU.
# Also, the default learning rate is not optimal for all of the models, so
# to achieve maximum accuracy it would be necessary to tune for each model
# separately.
# 

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
# criterion = FocalLossV1()


# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


# image_datasets_2 = {x: datasets.ImageFolder(os.path.join(data_dir_2, x), data_transforms[x]) for x in ['train', 'val']}
# # Create training and validation dataloaders
# dataloaders_dict_2 = {x: torch.utils.data.DataLoader(image_datasets_2[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}





'''
second train
'''

image_datasets_2 = image_datasets
dataloaders_dict_2 = dataloaders_dict
feature_extract = False
num_epochs = 50
# Detect if we have a GPU available
if not torch.cuda.is_available() == True:
    raise IOError('The cuda is not available!')
device = torch.device("cuda:0") #if torch.cuda.is_available() else "cpu"

params_to_update = model_ft.parameters()
set_parameter_requires_grad(model_ft, feature_extract)
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.Adam(params_to_update, lr=0.0001,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
model_ft, hist = train_model(model_ft, dataloaders_dict_2, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))



# '''
# third train
# '''
# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)
            
# optimizer_ft = optim.Adam(params_to_update, lr=0.00001,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
# model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=45, is_inception=(model_name=="inception"))


'''
forth
'''
# =============================================================================
# image_datasets_3 = {x: datasets.ImageFolder(os.path.join(data_dir_all, x), data_transforms[x]) for x in ['train', 'val']}
# # Create training and validation dataloaders
# dataloaders_dict_3 = {x: torch.utils.data.DataLoader(image_datasets_3[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
# 
# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)
#             
# optimizer_ft = optim.Adam(params_to_update, lr=0.00001,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
# model_ft, hist = train_model(model_ft, dataloaders_dict_3, criterion, optimizer_ft, num_epochs=120, is_inception=(model_name=="inception"))
# =============================================================================











'''
Evaluate
'''
# exVal_image_datasets = datasets.ImageFolder(exVal_dir, exVal_transforms) 
exVal_image_datasets = MyDataset(exVal_dir, exVal_transforms) 
# Create training and validation dataloaders
exVal_dataloaders_dict = torch.utils.data.DataLoader(exVal_image_datasets, batch_size=len(exVal_image_datasets), shuffle=False, num_workers=0) 
# random.seed(0)
exVal_acc,exVal_sens,exVal_spec = ExValidation(model_ft, exVal_dataloaders_dict, criterion)



'''
read output weights
'''

def readoutputs(model, exvalDataloaders):

    model.eval()   # Set model to evaluate mode
    out =[]
    gt_labels_list = []
    for inputs, labels in exvalDataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            outputs = model(inputs)
            outputs = outputs.data.cpu().detach()[0].numpy().tolist()
            gt_labels = labels.data.cpu().detach().numpy()
            out.append(outputs)
            gt_labels_list = np.concatenate((gt_labels_list, gt_labels))  
    return out, gt_labels_list

exVal_image_datasets1 = MyDataset(os.path.join(data_dir, 'train'), data_transforms['train'])
# Create training and validation dataloaders
exVal_dataloaders_dict1 = torch.utils.data.DataLoader(exVal_image_datasets1, batch_size= 1, shuffle=False, num_workers=0) 
train_scores, train_labels = readoutputs(model_ft, exVal_dataloaders_dict1)
train_scores = np.array(train_scores)
train_labels = np.array(train_labels).astype(int)
train_labels = train_labels[:,np.newaxis]
train_label_score  = np.concatenate((train_labels, train_scores),axis=1)
print(train_label_score.shape)


exVal_image_datasets2 = MyDataset(os.path.join(data_dir, 'val'), data_transforms['val'])
# Create training and validation dataloaders
exVal_dataloaders_dict2 = torch.utils.data.DataLoader(exVal_image_datasets2, batch_size= 1, shuffle=False, num_workers=0) 
val_scores, val_labels = readoutputs(model_ft, exVal_dataloaders_dict2)
val_scores = np.array(val_scores)
val_labels = np.array(val_labels).astype(int)
val_labels = val_labels[:,np.newaxis]
val_label_score  = np.concatenate((val_labels, val_scores),axis=1)
print(val_label_score.shape)


exVal_image_datasets3 = exVal_image_datasets
# Create training and validation dataloaders
exVal_dataloaders_dict3 = torch.utils.data.DataLoader(exVal_image_datasets3, batch_size= 1, shuffle=False, num_workers=0) 
test_scores, test_labels = readoutputs(model_ft, exVal_dataloaders_dict3)
test_scores = np.array(test_scores)
test_labels = np.array(test_labels).astype(int)
test_labels = test_labels[:,np.newaxis]
test_label_score  = np.concatenate((test_labels, test_scores),axis=1)
print(test_label_score.shape)

import pandas as pd
# df1 = pd.DataFrame(train_label_score,columns=['train_label','train_0','train_1'])
# df2 = pd.DataFrame(val_label_score,columns=['val_label','val_0','val_1'])
# df3 = pd.DataFrame(test_label_score,columns=['test_label','test_0','test_1'])
# df1.to_excel('train_'+ typeln +'.xls')
# df2.to_excel('val_'+ typeln +'.xls')
# df3.to_excel('test_'+ typeln +'.xls')

######################################################################
'''
 Initialize and Reshape the Networks
 -----------------------------------
 
 Now to the most interesting part. Here is where we handle the reshaping
 of each network. Note, this is not an automatic procedure and is unique
 to each model. Recall, the final layer of a CNN model, which is often
 times an FC layer, has the same number of nodes as the number of output
 classes in the dataset. Since all of the models have been pretrained on
 Imagenet, they all have output layers of size 1000, one node for each
 class. The goal here is to reshape the last layer to have the same
 number of inputs as before, AND to have the same number of outputs as
 the number of classes in the dataset. In the following sections we will
 discuss how to alter the architecture of each model individually. But
 first, there is one important detail regarding the difference between
 finetuning and feature-extraction.
 
 When feature extracting, we only want to update the parameters of the
 last layer, or in other words, we only want to update the parameters for
 the layer(s) we are reshaping. Therefore, we do not need to compute the
 gradients of the parameters that we are not changing, so for efficiency
 we set the .requires_grad attribute to False. This is important because
 by default, this attribute is set to True. Then, when we initialize the
 new layer and by default the new parameters have ``.requires_grad=True``
 so only the new layer’s parameters will be updated. When we are
 finetuning we can leave all of the .required_grad’s set to the default
 of True.
 
 Finally, notice that inception_v3 requires the input size to be
 (299,299), whereas all of the other models expect (224,224).
 
 Resnet
 ~~~~~~
 
 Resnet was introduced in the paper `Deep Residual Learning for Image
 Recognition <https://arxiv.org/abs/1512.03385>`__. There are several
 variants of different sizes, including Resnet18, Resnet34, Resnet50,
 Resnet101, and Resnet152, all of which are available from torchvision
 models. Here we use Resnet18, as our dataset is small and only has two
 classes. When we print the model, we see that the last layer is a fully
 connected layer as shown below:
 
 ::
 
    (fc): Linear(in_features=512, out_features=1000, bias=True) 
 
 Thus, we must reinitialize ``model.fc`` to be a Linear layer with 512
 input features and 2 output features with:
 
 ::
 
    model.fc = nn.Linear(512, num_classes)
 
 Alexnet
 ~~~~~~~
 
 Alexnet was introduced in the paper `ImageNet Classification with Deep
 Convolutional Neural
 Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__
 and was the first very successful CNN on the ImageNet dataset. When we
 print the model architecture, we see the model output comes from the 6th
 layer of the classifier
 
 ::
 
    (classifier): Sequential(
        ...
        (6): Linear(in_features=4096, out_features=1000, bias=True)
     ) 
 
 To use the model with our dataset we reinitialize this layer as
 
 ::
 
    model.classifier[6] = nn.Linear(4096,num_classes)
 
 VGG
 ~~~
 
 VGG was introduced in the paper `Very Deep Convolutional Networks for
 Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`__.
 Torchvision offers eight versions of VGG with various lengths and some
 that have batch normalizations layers. Here we use VGG-11 with batch
 normalization. The output layer is similar to Alexnet, i.e.
 
 ::
 
    (classifier): Sequential(
        ...
        (6): Linear(in_features=4096, out_features=1000, bias=True)
     )
 
 Therefore, we use the same technique to modify the output layer
 
 ::
 
    model.classifier[6] = nn.Linear(4096,num_classes)
 
 Squeezenet
 ~~~~~~~~~~
 
 The Squeeznet architecture is described in the paper `SqueezeNet:
 AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
 size <https://arxiv.org/abs/1602.07360>`__ and uses a different output
 structure than any of the other models shown here. Torchvision has two
 versions of Squeezenet, we use version 1.0. The output comes from a 1x1
 convolutional layer which is the 1st layer of the classifier:
 
 ::
 
    (classifier): Sequential(
        (0): Dropout(p=0.5)
        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
        (2): ReLU(inplace)
        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
     ) 
 
 To modify the network, we reinitialize the Conv2d layer to have an
 output feature map of depth 2 as
 
 ::
 
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
 
 Densenet
 ~~~~~~~~
 
 Densenet was introduced in the paper `Densely Connected Convolutional
 Networks <https://arxiv.org/abs/1608.06993>`__. Torchvision has four
 variants of Densenet but here we only use Densenet-121. The output layer
 is a linear layer with 1024 input features:
 
 ::
 
    (classifier): Linear(in_features=1024, out_features=1000, bias=True) 
 
 To reshape the network, we reinitialize the classifier’s linear layer as
 
 ::
 
    model.classifier = nn.Linear(1024, num_classes)
 
 Inception v3
 ~~~~~~~~~~~~
 
 Finally, Inception v3 was first described in `Rethinking the Inception
 Architecture for Computer
 Vision <https://arxiv.org/pdf/1512.00567v1.pdf>`__. This network is
 unique because it has two output layers when training. The second output
 is known as an auxiliary output and is contained in the AuxLogits part
 of the network. The primary output is a linear layer at the end of the
 network. Note, when testing we only consider the primary output. The
 auxiliary output and primary output of the loaded model are printed as:
 
 ::
 
    (AuxLogits): InceptionAux(
        ...
        (fc): Linear(in_features=768, out_features=1000, bias=True)
     )
     ...
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
 
 To finetune this model we must reshape both layers. This is accomplished
 with the following
 
 ::
 
    model.AuxLogits.fc = nn.Linear(768, num_classes)
    model.fc = nn.Linear(2048, num_classes)
 
 Notice, many of the models have similar output structures, but each must
 be handled slightly differently. Also, check out the printed model
 architecture of the reshaped network and make sure the number of output
 features is the same as the number of classes in the dataset.
'''










'''
######################################################################
# Comparison with Model Trained from Scratch
# ------------------------------------------Best val Acc: 0.792342 with sens 0.639110 and spec 0.852923
# 
# Just for fun, lets see how the model learns if we do not use transfer
# learning. The performance of finetuning vs. feature extracting depends
# largely on the dataset but in general both transfer learning methods
# produce favorable results in terms of training time and overall accuracy
# versus a model trained from scratch.
# 

# Initialize the non-pretrained version of the model used for this run
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Plot the training curves of validation accuracy vs. number 
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


######################################################################
# Final Thoughts and Where to Go Next
# -----------------------------------
# 
# Try running some of the other models and see how good the accuracy gets.
# Also, notice that feature extracting takes less time because in the
# backward pass we do not have to calculate most of the gradients. There
# are many places to go from here. You could:
# 
# -  Run this code with a harder dataset and see some more benefits of
#    transfer learning
# -  Using the methods described here, use transfer learning to update a
#    different model, perhaps in a new domain (i.e. NLP, audio, etc.)
# -  Once you are happy with a model, you can export it as an ONNX model,
#    or trace it using the hybrid frontend for more speed and optimization
#    opportunities.
# 
'''
