#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt

def readLabels(label_file_path='')->list:

    labels_file = open(label_file_path, 'rb')
    
    # 获取魔数
    magic_number = int.from_bytes(labels_file.read(4), byteorder='big')

    # 获取 label_number
    label_number =int.from_bytes(labels_file.read(4), byteorder='big')


    # 获取 labels
    labels = []
    for i in range(label_number):
        temp_label = int.from_bytes(labels_file.read(1), byteorder='big')
        labels.append(temp_label)
    labels_file.close()

    # data = labels_file.read()
    # print(data[0])

    return label_number, labels

def show(images_file_path='')->None:
    images_file = open(images_file_path, 'rb')
    
    magic_number = int.from_bytes(images_file.read(4), byteorder='big')
    images_number = int.from_bytes(images_file.read(4), byteorder='big')
    rows_number = int.from_bytes(images_file.read(4), byteorder='big')
    columns_number = int.from_bytes(images_file.read(4), byteorder='big')
    print(magic_number, images_number, rows_number, columns_number)
    data = images_file.read()

    print(type(data))
    print(len(data))
    
    data = list(bytes(data))
    data = torch.tensor(data)
    print(type(data))

    # print(data[0])
    # print(data[1])
    # print(data[2])
    # print(data[3])
    # 显示第一个 image
    for i in range(rows_number):
        for j in range(columns_number):
            # pixel_val = int.from_bytes(images_file.read(1), byteorder='big')
            pixel_val = data[i*28+j]
            if(pixel_val==0):
                print('.', end="")
            else:
                print('*', end='')
        print("")
    images_file.close()

    pass

if __name__ == '__main__':
    train_images_path = './data/'
    train_labels_path = './data/train-labels-idx1-ubyte'
    test_images_path = './data/t10k-images-idx3-ubyte'
    test_labels_path = './data/t10k-labels-idx1-ubyte'

    test_labels_number, test_labels = readLabels(test_labels_path)

    # print("test_labels_numer:", test_labels_number)
    # print("test_labels:", test_labels)

    show(test_images_path)
