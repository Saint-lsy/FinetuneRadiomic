from torch.utils.data import Dataset
import numpy as np
import glob
import cv2
import os
import re
##一张病灶灰度转RGB
class MyDataset2(Dataset):
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
                self.datas.append(img[:,:,i])
                self.labels.append(label)
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img = self.datas[index] 
        label= int(self.labels[index])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform is not None:
            img = self.transform(img) 
        return img, label              #这就返回一个样本
    
    def __len__(self):
        return len(self.datas)          #返回长度，index就会自动的指导读取多少



##三张病灶结合成三通道
class MyDataset(Dataset):
    def __init__(self, file_path, transform = None, target_transform = None):
        """
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.img_paths = []
        self.imgs_name = os.listdir(file_path)
        self.imgs_name.sort(key = lambda i:int(re.match(r'\d+',i.strip('info_')).group()))  
        self.img_paths= list(map(lambda x: os.path.join(file_path, x), self.imgs_name))     
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