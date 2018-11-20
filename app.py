# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:16:57 2018

@author: zhl
"""
import tensorflow as tf
import forward
import backward

import debug
import numpy as np
import cv2

DEBUG_PRINT = True
BATCH_SIZE = 1
SAMPLE_NUM = 1
STEP = 1

Debug = debug.DebugTool(True)

def Test(mnist):
     
    with tf.Graph().as_default() as g:
        
        x = tf.placeholder(tf.float32,[BATCH_SIZE,forward.IMAGE_WIDTH,
                                       forward.IMAGE_HEIGTH,forward.NUM_CHANNELS])
    
        y = forward.forward(x, False)
        saver = tf.train.Saver()       
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
#            print(ckpt)
#            print(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:               
                    reshape = np.reshape(mnist,[BATCH_SIZE,forward.IMAGE_WIDTH,
                                           forward.IMAGE_HEIGTH,forward.NUM_CHANNELS])

                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score = sess.run(y,feed_dict={x:reshape})
#                    print("准确率 %g" % accuracy_score)
                    print(accuracy_score)
            else:
                print('No checkpoint file found')
                return   

def main():
    
    src = cv2.imread("./app_image/4.jpg")
    src = cv2.resize(src,(250,250), interpolation=cv2.INTER_LINEAR)           #将读出的图片转换成指定大小
#    src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
#    src = cv2.equalizeHist(src) 
    Test(src)
    
if __name__ == "__main__":
    main()
    