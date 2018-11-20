# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:16:08 2018

@author: zhl
"""

import tensorflow as tf 
import forward
import vggMakeData
import numpy as np
import os
import debug
import cv2

Debug = debug.DebugTool(False) #定义一个调试工具

BATCH_SIZE = 10
STEP = 1000
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "face_model"

def backward():
    Debug.print_error()
    X = tf.placeholder(tf.float32,shape=[BATCH_SIZE,
                                         forward.IMAGE_WIDTH,
                                         forward.IMAGE_HEIGTH,
                                         forward.NUM_CHANNELS])
    Debug.print_error()
    y_real = tf.placeholder(tf.float32,
                            shape=[None,forward.OUTPUT_NODE])       #X对应的真实标签
    y_predict = forward.forward(X, True)                            #X对应的预测值
    print("前向传播网络搭建完毕")
   
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(y_real,1),logits=y_predict)

    with tf.name_scope("train_op"):    
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        
    saver = tf.train.Saver()
    
    Debug.print_error()
    
    #获取数据集
    image_data = vggMakeData.ImageData(vggMakeData.tfRecorf_path, BATCH_SIZE)
    image_data.read_tfRecord()
    img_batch,label_batch = image_data.get_data()
    
    #创建会话sess
    Debug.print_error()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for index in range(STEPS):

            x_data, y_data = sess.run([img_batch,label_batch])
            
            Debug.print_current_value(y_data)
            reshape_xs = np.reshape(x_data,[BATCH_SIZE,
                                            forward.IMAGE_WIDTH,
                                            forward.IMAGE_HEIGTH,
                                            forward.NUM_CHANNELS])
            Debug.print_error()
            _, loss_value = sess.run([train_op, loss], 
                                           feed_dict={X:reshape_xs, y_real:y_data})
            
            #每训练100轮保存当前的模型
            if (index) % 20 == 0:
                print("After %d training steps,loss on training batch is %g." % (index, loss_value))
                #把当前会话保存到指定的路径下面
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                           global_step=index)
        
        coord.request_stop()
        coord.join(threads)
        

def testData():
    '''function:测试读出的数据格式'''    
    image_data = vggMakeData.ImageData(vggMakeData.tfRecorf_path, 1)
    image_data.read_tfRecord()
    img_batch,label_batch = image_data.get_data()
    
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    
    x_data, y_data = sess.run([img_batch,label_batch])
    print(y_data)
    x_data = (x_data + 1) * 127.5
    print(len(x_data))
    x_reshape = np.reshape(x_data, (250, 250, 3))
    print(x_reshape)
    dst_name = "./a.jpg"
    cv2.imwrite(dst_name,x_reshape)

    coord.request_stop()
    coord.join(threads)


def main():
    backward()
#    testData()
    
if __name__ == "__main__":
    main()    
    
    