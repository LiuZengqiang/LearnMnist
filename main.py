# TODO: 实现 LeNet
from typing import Iterable, Optional, Sequence, Union
import torch 
import numpy as np
import torch.nn as nn

from torch import Tensor    # 基本的计算单位

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
import torch.nn.functional as nnf
import os
import matplotlib.pyplot as plt

# LeNet 网络模型
class LeNet5(nn.Module):
    """
    Input: 1*32*32
    Output: 10
    """
    def __init__(self) -> None:
        super(LeNet5,self).__init__()
        # 卷积层
        # 输入通道数 1, 输出通道数(卷积核组数) 6, 卷积核尺寸 5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 池化层 下采样层
        # 最大池化层 池化核大小 2
        self.pool1 = nn.MaxPool2d(2)
        # 卷积
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 池化
        self.pool2 = nn.MaxPool2d(2)
        # 全连接
        self.fc1 = nn.Linear(400, 120)
        # 全连接
        self.fc2 = nn.Linear(120, 84)
        # 全连接
        self.fc3 = nn.Linear(84, 10)
        # 加一个 relu
        # self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x

"""
x: Tensor
y: int
"""
class LeNetDataSet(Dataset):
    def __init__(self, images_file_path, labels_file_path) -> None:
        super().__init__()
        # 获取 images 数据 -> x
        images_file = open(images_file_path, 'rb')
        magic_number = int.from_bytes(images_file.read(4), byteorder='big')
        images_number = int.from_bytes(images_file.read(4), byteorder='big')
        rows_number = int.from_bytes(images_file.read(4), byteorder='big')
        columns_number = int.from_bytes(images_file.read(4), byteorder='big')
        # print(magic_number, images_number, rows_number, columns_number)
        
        data = torch.tensor(list(bytes(images_file.read())), dtype=float)

        data = data.reshape(images_number, rows_number, columns_number)
        
        x = torch.zeros([images_number, 1, 32, 32], dtype=float)
        
        # for img_idx in range(images_number):
        #     for row_idx in range(rows_number):
        #         for col_idx in range(columns_number):
        #             pixel_val = int.from_bytes(images_file.read(1), byteorder='big')
        #             x[img_idx][row_idx][col_idx] = pixel_val
        images_file.close()
        
        # 将 x 变为 32*32
        for img_idx in range(images_number):
            temp = data[img_idx]
            # print(temp)
            temp = temp.reshape(1,1,rows_number, columns_number)
            temp = nnf.interpolate(temp, size=(32, 32), mode='bilinear', align_corners=False)
            x[img_idx][0] = temp
        
        self.x = x

        # 读取 labels 数据
        labels_file = open(labels_file_path, 'rb')
        # 获取魔数
        magic_number = int.from_bytes(labels_file.read(4), byteorder='big')
        # 获取 label_number
        label_number =int.from_bytes(labels_file.read(4), byteorder='big')
        # 获取 labels
        y = torch.zeros([label_number], dtype=float)
        for idx in range(label_number):
            temp_label = int.from_bytes(labels_file.read(1), byteorder='big')
            y[idx] = temp_label
        labels_file.close()
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return len(self.x)


if __name__ == '__main__':

    train_images_path = './data/train-images-idx3-ubyte'
    train_labels_path = './data/train-labels-idx1-ubyte'
    test_images_path = './data/t10k-images-idx3-ubyte'
    test_labels_path = './data/t10k-labels-idx1-ubyte'
    batch_size = 256
    all_epoch = 100
    learn_rate = 1e-1
    
    # 构造 数据集
    print("Begin:: dataset")
    train_dataset = LeNetDataSet(train_images_path, train_labels_path)
    test_dataset = LeNetDataSet(test_images_path, test_labels_path)
    print("End:: dataset")
    
    print("Begin:: dataloader")
    # 构造 数据集加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("End:: dataloader")
    
    print("Begin:: Net")
    # 初始化 网络模型
    model = LeNet5()
    # 优化器
    # 说明优化的时候 要优化哪里的参数
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    l = []

    # 训练
    for epoch_idx in range(all_epoch):
        print(epoch_idx)
        model = model.train()
        for idx, (train_x, train_y) in enumerate(train_loader):
            
            # 先清空 优化器的梯度值
            optimizer.zero_grad()
            # 计算预测的y 
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_y.long())

            loss.backward()
            
            optimizer.step()
        with torch.no_grad():
            model = model.eval()
            # 从 test 中获取一组batch

            output = model(test_dataset.x.float())

            acc = (output.argmax(axis=1) == test_dataset.y.squeeze()).sum().item() / len(test_dataset)
            l.append(float(acc)) 
            print("Epoch: %d, Accuracy: %.3f" % (epoch_idx + 1, float(acc)))


    x = np.arange(1, len(l)+1)
    y = l
    plt.plot(x, y)
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), "weigth.pth")
    
