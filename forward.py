# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:15:36 2018

@author: zhl
"""

import tensorflow as tf

#定义神经网络可以接受的图片的尺寸和通道数
IMAGE_WIDTH = 250
IMAGE_HEIGTH = 250
NUM_CHANNELS = 3

#卷积核的大小
KERNEL_SIZE = (3, 3)

#第一层输出通道数
CONV_KERNEL_NUM_11 = 64
#第二层输出通道数
CONV_KERNEL_NUM_21 = 128
#第三层输出通道数
CONV_KERNEL_NUM_31 = 256
#第四层输出通道数
CONV_KERNEL_NUM_41 = 512
#第五层输出通道数
CONV_KERNEL_NUM_51 = 512
#全连接层：第一层神经元个数
FC_NODE_NUM_1 = 1024
#全连接层：第二层神经元个数
FC_NODE_NUM_2 = 1024
#输出层神经元的个数
OUTPUT_NODE = 2

def conv2d(x_image, output_channel, kernel_size, conv_name):
    '''function:卷积操作
    Arg:
        - x_image:输入图像;eg：[batch_size, width, heigth, channel_num]
        - output_channel:输出通道数
        - kernel_size:卷积核的大小
        - conv_name:卷积层的名称
        '''
    conv = tf.layers.conv2d(x_image,
                            output_channel,
                            kernel_size,
                            padding = 'same',
                            activation = tf.nn.relu,
                            name = conv_name)
    return conv
    
def max_pool_2x2(x, pool_name):
    '''function:
        Arg:
            - x:输入数据
            - pool_name:池化层的名称
    '''
    pooling = tf.layers.max_pooling2d(x,
                                      (2,2),
                                      (2,2),
                                      name = pool_name)
    return pooling

#前向传播过程，返回预测值prediction
def forward(x_image, train):
    '''function:搭建前向传播网络：卷积 + 全连接
        Arg：
        - x_image:输入的图像;eg：[batch_size, width, heigth, channel_num]
        - train:波尔型，eg：train=True->使用dropout
    '''
    #第一层卷积-1
    conv1_1 = conv2d(x_image, CONV_KERNEL_NUM_11, KERNEL_SIZE, "conv1_1")
    conv1_2 = conv2d(conv1_1, CONV_KERNEL_NUM_11, KERNEL_SIZE, "conv1_2")
    #池化
    pooling1 = max_pool_2x2(conv1_2, "pool1")
    
    #第二层卷积-1
    conv2_1 = conv2d(pooling1, CONV_KERNEL_NUM_21, KERNEL_SIZE, "conv2_1")
    conv2_2 = conv2d(conv2_1, CONV_KERNEL_NUM_21, KERNEL_SIZE, "conv2_2")
    #池化
    pooling2 = max_pool_2x2(conv2_2, "pool2")
    
    #第三层卷积-1
    conv3_1 = conv2d(pooling2, CONV_KERNEL_NUM_31, KERNEL_SIZE, "conv3_1")
    conv3_2 = conv2d(conv3_1, CONV_KERNEL_NUM_31, KERNEL_SIZE, "conv3_2")
    #池化
    pooling3 = max_pool_2x2(conv3_2,'pool3')   
    
    #图片的三维数据拉直【长*宽*通道数】
    flatten = tf.layers.flatten(pooling3)
    
    #全连接第一层
    fc_1 = tf.layers.dense(flatten,
                           FC_NODE_NUM_1,
                           activation = tf.nn.relu,
                           name = "fc_1")
    if train:
        fc_1 = tf.nn.dropout(fc_1, 0.5)
    
    predict = tf.layers.dense(fc_1,
                          OUTPUT_NODE,
                          activation = None,
                          name = "fc_2")
    return predict
