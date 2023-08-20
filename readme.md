## LearnMnist
### requirment:
```
torch
```
### How to implement the LeNet-5 form zero?
1. 下载训练、测试数据集  
    >> 本仓库已经将训练、测试数据集下载并解压好放到data目录下了  

    如果你参考本仓库从0开始实现，需要去[THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)中下载 ``train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz`` 这四个文件放到``data``目录下，并使用以下命令将文件解压：  
    ```shell
    # 在 data 目录下
    gzip ./train*
    gzip ./t10*
    ```

2. 设计网络模型，编写python代码  
代码见文件``main.py``。  
``main.py``中包含了``LeNet-5``网络的结构、数据集加载、模型训练、模型测试和保存训练完成后模型参数值的代码。  

3. 运行``main.py``脚本：训练+测试+保存
    ```shell
    python3 main.py
    ```