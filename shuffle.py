
import os
import numpy as np
import shutil
#from sklearn.utils import shuffle

all_data = '/data/radiomic/tumor_V/'    #源数据集图像的文件夹的路径
target_dir = '/data/radiomic/Tumor_V/'
# rootdir1 = ''    #源数据集标签的文件夹的路径
def shuffle(rootdir, targetdir):
    a = os.listdir(rootdir)
    np.random.seed(3)
    np.random.shuffle(a)    #将数据集打乱顺序
    
    d = int(len(a)*8/10)    #将数据集分为两部分，在这里可以根据自己的需要修改
    b = a[:d]    #数据集的前半部分
    c = a[d:]    #数据集的后半部分
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)     
    traindir = os.path.join(targetdir,'train')    
    valdir = os.path.join(targetdir,'val')  
    if not os.path.exists(traindir):
        os.mkdir(traindir)    #新建文件夹以保存随机数据集
    if not os.path.exists(valdir): 
        os.mkdir(valdir)   #新建文件夹以保存随机数据集的图片部分
    # os.mkdir(os.path.join(''))    #新建文件夹以保存随机数据集的标签部分
    # os.mkdir(os.path.join(''))    #新建文件夹以保存随机数据集
    # os.mkdir(os.path.join(''))    #新建文件夹以保存随机数据集的图片部分
    # os.mkdir(os.path.join(''))    #新建文件夹以保存随机数据集的标签部分
    for i in b:
        tragetpic_dir_1 = os.path.join(traindir, i)    #随机数据集的图像的路径
        # targetlab_dir_1 = os.path.join('', i)    #随机数据集的标签的路径
        oripic_dir_1 = os.path.join(rootdir, i)       #原始数据集的图像的路径
        # orilab_dir_1 = os.path.join('', i)       #原始数据集的标签的路径
        shutil.copy(oripic_dir_1, tragetpic_dir_1)
        # shutil.copy(orilab_dir_1, tragetlab_dir_1)
    
    for j in c:
        tragetpic_dir_2 = os.path.join(valdir, j)    #随机数据集的图像的路径
        # targetlab_dir_2 = os.path.join('', j)    #随机数据集的标签的路径
        oripic_dir_2 = os.path.join(rootdir, j)       #原始数据集的图像的路径
        # orilab_dir_2 = os.path.join('', j)       #原始数据集的标签的路径
        shutil.copy(oripic_dir_2, tragetpic_dir_2)
        # shutil.copy(orilab_dir_2, tragetlab_dir_2)
shuffle(all_data, target_dir)