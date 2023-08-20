import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as nnf

# LeNet5 网络模型
class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5,self).__init__()
        # 卷积层 conv1：输入通道数:1, 卷积核组数:6, 卷积核尺寸:5*5
        # 输入: batch*1*32*32
        # 输出: batch*6*28*28
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 最大池化层 maxpool：池化核尺寸:2*2
        # 输入：batch*6*28*28
        # 输出：batch*6*14*14
        self.pool1 = nn.MaxPool2d(2)
        # 卷积层 conv2：输入通道数:6, 卷积核组数:16, 卷积核尺寸:5*5
        # 输入：batch*6*14*14
        # 输出：batch*16*10*10
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 最大池化层 maxpool: 池化核尺寸:2*2
        # 输入：batch*16*10*10
        # 输出：batch*16*5*5
        self.pool2 = nn.MaxPool2d(2)
        # 全连接层（在该层之前需要将每个batch的数据展开成一维）
        # 输入：batch*400 (其中的400=16*5*5)
        # 输出：btahc*120
        self.fc1 = nn.Linear(400, 120)
        # 全连接层
        # 输入：batch*120
        # 输出：batch*84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层
        # 输入：batch*84
        # 输出：batch*10
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # 将每个batch的数据展开为一维
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

# 数据集 类，用于存储需要用到的训练、测试数据
class LeNetDataSet(Dataset):
    def __init__(self, images_file_path, labels_file_path) -> None:
        super().__init__()
        
        # 获取 images 原始数据, 按照字节读取文件
        images_file = open(images_file_path, 'rb')
        # images_file 的前4*4个字节分别表示 魔数, 图片个数, 每张图片像素行数(高), 每张图片像素列数(宽)
        magic_number = int.from_bytes(images_file.read(4), byteorder='big')
        images_number = int.from_bytes(images_file.read(4), byteorder='big')
        rows_number = int.from_bytes(images_file.read(4), byteorder='big')
        columns_number = int.from_bytes(images_file.read(4), byteorder='big')

        # 读取images数据, images在文件中测存储格式可以参考 [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)
        images_data = torch.tensor([bytes(images_file.read())], dtype=float)        
        images_file.close()
        
        # 将 images_data 的维度变为 images_number * 1 * rows_number * columns_number, (NCHW)
        images_data = images_data.reshape(images_number, 1, rows_number, columns_number)
        # 将每张image的大小调整为 32*32
        images_data = nnf.interpolate(images_data, size=(32, 32), mode='bilinear', align_corners=False)
        # 将每张image中的像素值调整到 [0.0, 1.0] 之间
        images_data = (images_data-0.0)/(255.0-0.0)
        self.x = images_data

        # 读取label数据, label数据在文件中测存储格式可以参考 [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)
        labels_file = open(labels_file_path, 'rb')
        magic_number = int.from_bytes(labels_file.read(4), byteorder='big')
        labels_number =int.from_bytes(labels_file.read(4), byteorder='big')
        labels_data = torch.tensor([bytes(labels_file.read())], dtype=int)
        labels_data = labels_data.reshape(labels_number)
        labels_file.close()
        self.y = labels_data

    # 因为继承了Dataset类, 因此需要重载 __getitem__() 和 __len__() 函数
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return len(self.x)

if __name__ == '__main__':
    
    # 训练数据 image 文件路径
    train_images_path = './data/train-images-idx3-ubyte'
    # 训练数据 label 文件路径
    train_labels_path = './data/train-labels-idx1-ubyte'
    # 测试数据 image 文件路径
    test_images_path = './data/t10k-images-idx3-ubyte'
    # 测试数据 label 文件路径
    test_labels_path = './data/t10k-labels-idx1-ubyte'

    batch_size = 256    # 每个训练 epoch 的输入 batch 数
    all_epoch = 100     # 总的训练 epoch
    learn_rate = 1e-1   # 学习率
    accuracy_threshold = 1e-3
    pre_acc = 0.0

    # 构造 进行训练、测试的 数据集对象
    print("Begin:: dataset")
    train_dataset = LeNetDataSet(train_images_path, train_labels_path)
    test_dataset = LeNetDataSet(test_images_path, test_labels_path)
    print("End:: dataset")

    # 构造 数据集加载器
    # 用于将训练、测试的数据输送给模型 model
    print("Begin:: dataloader")
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("End:: dataloader")
    
    # 初始化 网络模型
    model = LeNet5()
    # 优化器
    # 选择使用 SGD 优化器 优化 model里的参数(model.parameters()), 学习率设为 learn_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    # 损失函数
    # 使用 交叉熵 作为损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = []   # 用于记录每次 epoch 后模型对test数据集的精度

    # 训练
    print("Begin:: train")
    for epoch_idx in range(all_epoch):
        # 将 model 设置为 train 模式(训练模式)
        model.train()
        # 每次从 训练集合中取出一部分数据进行训练, 
        # 每次训练的数据维度为 train_x.shape={batch_size,1,32,32}, train_y.shape={batch_size,1}
        for idx, (train_x, train_y) in enumerate(train_loader):
            # 先清空 优化器的梯度值
            optimizer.zero_grad()
            # 使用 model 计算对 train_x 的预测结果 predict_y
            predict_y = model(train_x.float())
            
            # 计算 predict_y 和 真值 train_y 的损失函数值(差异程度)
            # predict_y 是一个大小为 batch_size*10 的 tensor, 
            # 每个batch中的10个数值分别表示该图片是数字{0,1,2..,9}的可能性, 数值越大表示可能性越高
            # train_y 是一个大小为 batch_size*1 的 tensor, 
            # 每个batch中的1个数值表示该图片的真实label
            loss = loss_fn(predict_y, train_y.long())
            # 根据 交叉熵 损失值对模型(网络)进行反向传播，计算每个参数对于loss的梯度
            loss.backward()
            # 使用前面定义的优化器 optimizer 对 model 中的参数进行优化调整
            optimizer.step()

        with torch.no_grad():
            # 将 model 设置为 eval 模式(推理模式)
            model.eval()
            # 预测 test 集合中的 image 的结果 predict_output
            predict_output = model(test_dataset.x.float())
            # 计算 预测的精度, 如果预测的都正确 acc = 100%
            acc = (predict_output.argmax(axis=1) == test_dataset.y.squeeze()).sum().item() / len(test_dataset)
            accuracy.append(float(acc)) 
            print("Epoch: %d, Accuracy: %.3f" % (epoch_idx + 1, float(acc)))
    print("End:: train")
    # 保存模型
    # 将训练完成后的模型参数值保存在文件中
    # 以后可以通过 model.state_dict(torch.load('weights.pth')) 直接加载训练好的模型
    torch.save(model.state_dict(), "weigths.pth")