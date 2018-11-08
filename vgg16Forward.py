
import tensorflow as tf

#定义神经网络可以接受的图片的尺寸和通道数
IMAGE_WIDTH = 128
IMAGE_HEIGTH = 96
NUM_CHANNELS = 1

#卷积核的大小
CONV_SIZE = 3

#第一层卷积核的个数【重复两次】
CONV_KERNEL_NUM_11 = 64
CONV_KERNEL_NUM_12 = 64

#第二层卷积核的个数【重复两次】
CONV_KERNEL_NUM_21 = 128
CONV_KERNEL_NUM_22 = 128

#第三层卷积核的个数【重复三次】
CONV_KERNEL_NUM_31 = 256
CONV_KERNEL_NUM_32 = 256
CONV_KERNEL_NUM_33 = 256

#第四层卷积核的个数【重复三次】
CONV_KERNEL_NUM_41 = 512
CONV_KERNEL_NUM_42 = 512
CONV_KERNEL_NUM_43 = 512

#第五层卷积核的个数【重复三次】
CONV_KERNEL_NUM_51 = 512
CONV_KERNEL_NUM_52 = 512
CONV_KERNEL_NUM_53 = 512

#全连接层：第一层神经元个数
FC_NODE_NUM_1 = 4096
#全连接层：第二层神经元个数
FC_NODE_NUM_2 = 4096
#输出层神经元的个数
FC_OUTPUT_NODE = 2

def weight_variable(shape,regularizer):
    #随机生成权重
    initial = tf.truncated_normal(shape,stddev=0.01)
    #正则化操作
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(initial))
    return tf.Variable(initial)

def bias_variable(shape):
    #生成偏重的值为0.1
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def kernel_variable(shape):
    #随机生成卷积核
    initial = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def conv2d(x, kernel):
    #卷积操作，x:要训练的数据；w:卷积核的大小
    return tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding="SAME")


def max_pool_2x2(x):
    #最大池化操作
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#前向传播过程，返回预测值prediction
def forward(x, train, regularizer):
    
    #第一层卷积-1
    kernel_conv_11 = kernel_variable([CONV_SIZE,CONV_SIZE,NUM_CHANNELS,CONV_KERNEL_NUM_11])
    b_conv_11 = bias_variable([CONV_KERNEL_NUM_11])
    out_conv_11 = tf.nn.relu(conv2d(x,kernel_conv_11) + b_conv_11)
    #第一层卷积-2
    kernel_conv_12 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_11,CONV_KERNEL_NUM_12])
    b_conv_12 = bias_variable([CONV_KERNEL_NUM_12])
    out_conv_12 = tf.nn.relu(conv2d(out_conv_11,kernel_conv_12) + b_conv_12)
    #池化
    pool_conv_1 = max_pool_2x2(out_conv_12)
    
    #第二层卷积-1
    kernel_conv_21 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_12,CONV_KERNEL_NUM_21])
    b_conv_21 = bias_variable([CONV_KERNEL_NUM_21])
    out_conv_21 = tf.nn.relu(conv2d(pool_conv_1,kernel_conv_21) + b_conv_21)
    #第二层卷积-2
    kernel_conv_22 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_21,CONV_KERNEL_NUM_22])
    b_conv_22 = bias_variable([CONV_KERNEL_NUM_22])
    out_conv_22 = tf.nn.relu(conv2d(out_conv_21,kernel_conv_22) + b_conv_22)
    #池化
    pool_conv_2 = max_pool_2x2(out_conv_22)
    
    #第三层卷积-1
    kernel_conv_31 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_22,CONV_KERNEL_NUM_31])
    b_conv_31 = bias_variable([CONV_KERNEL_NUM_31])
    out_conv_31 = tf.nn.relu(conv2d(pool_conv_2,kernel_conv_31) + b_conv_31)
    #第三层卷积-2
    kernel_conv_32 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_31,CONV_KERNEL_NUM_32])
    b_conv_32 = bias_variable([CONV_KERNEL_NUM_32])
    out_conv_32 = tf.nn.relu(conv2d(out_conv_31,kernel_conv_32) + b_conv_32)
    #第三层卷积-3
    kernel_conv_33 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_32,CONV_KERNEL_NUM_33])
    b_conv_33 = bias_variable([CONV_KERNEL_NUM_33])
    out_conv_33 = tf.nn.relu(conv2d(out_conv_32,kernel_conv_33) + b_conv_33)
    #池化
    pool_conv_3 = max_pool_2x2(out_conv_33)
    
    #第四层卷积-4
    kernel_conv_41 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_33,CONV_KERNEL_NUM_41])
    b_conv_41 = bias_variable([CONV_KERNEL_NUM_41])
    out_conv_41 = tf.nn.relu(conv2d(pool_conv_3,kernel_conv_41) + b_conv_41)
    #第四层卷积-2
    kernel_conv_42 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_41,CONV_KERNEL_NUM_42])
    b_conv_42 = bias_variable([CONV_KERNEL_NUM_42])
    out_conv_42 = tf.nn.relu(conv2d(out_conv_41,kernel_conv_42) + b_conv_42)
    #第四层卷积-3
    kernel_conv_43 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_42,CONV_KERNEL_NUM_43])
    b_conv_43 = bias_variable([CONV_KERNEL_NUM_43])
    out_conv_43 = tf.nn.relu(conv2d(out_conv_42,kernel_conv_43) + b_conv_43)
    #池化
    pool_conv_4 = max_pool_2x2(out_conv_43)
    
    #第五层卷积-4
    kernel_conv_51 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_43,CONV_KERNEL_NUM_51])
    b_conv_51 = bias_variable([CONV_KERNEL_NUM_51])
    out_conv_51 = tf.nn.relu(conv2d(pool_conv_4,kernel_conv_51) + b_conv_51)
    #第五层卷积-2
    kernel_conv_52 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_51,CONV_KERNEL_NUM_52])
    b_conv_52 = bias_variable([CONV_KERNEL_NUM_52])
    out_conv_52 = tf.nn.relu(conv2d(out_conv_51,kernel_conv_52) + b_conv_52)
    #第五层卷积-3
    kernel_conv_53 = kernel_variable([CONV_SIZE,CONV_SIZE,CONV_KERNEL_NUM_52,CONV_KERNEL_NUM_53])
    b_conv_53 = bias_variable([CONV_KERNEL_NUM_53])
    out_conv_53 = tf.nn.relu(conv2d(out_conv_52,kernel_conv_53) + b_conv_53)
    #池化
    pool_conv_5 = max_pool_2x2(out_conv_53)
    
    #图片的三维数据拉直【长*宽*通道数】
    h_pool3_shape = pool_conv_5.get_shape().as_list()
    node = h_pool3_shape[1] * h_pool3_shape[2] * h_pool3_shape[3]
    input_fc = tf.reshape(pool_conv_5,[h_pool3_shape[0],node]) #喂入全连接层的数据
    
    #全连接第一层
    w_layer_1 = weight_variable([node,FC_NODE_NUM_1], regularizer)
    b_layer_1 = bias_variable([FC_NODE_NUM_1])
    out_layer_1 = tf.nn.relu(tf.matmul(input_fc,w_layer_1) + b_layer_1)
    if train:
        out_layer_1 = tf.nn.dropout(out_layer_1, 0.5)
    
    #全连接第二层
    w_layer_2 = weight_variable([FC_NODE_NUM_1,FC_NODE_NUM_2],regularizer)
    b_layer_2 = bias_variable([FC_NODE_NUM_2])
    out_layer_2 = tf.nn.relu(tf.matmul(out_layer_1,w_layer_2) + b_layer_2)
    if train:
        out_layer_2 = tf.nn.dropout(out_layer_2, 0.5)
    
    #全连接层第三层【输出层】
    w_layer_3 = weight_variable([FC_NODE_NUM_2,FC_OUTPUT_NODE],regularizer)
    b_layer_2 = bias_variable([FC_OUTPUT_NODE])
    predict = tf.matmul(out_layer_2,w_layer_3) + b_layer_2  #预测值
    
    return predict