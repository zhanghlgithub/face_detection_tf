# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:17:07 2018

@author: zhl
"""

import tensorflow as tf
import forward
import backward
import vggMakeData
import debug
import numpy as np

DEBUG_PRINT = True
BATCH_SIZE = 10
STEP = 100

Debug = debug.DebugTool(True)

def Test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[BATCH_SIZE,
                                       forward.IMAGE_WIDTH,
                                       forward.IMAGE_HEIGTH,
                                       forward.NUM_CHANNELS])
    
        y = forward.forward(x,False)
        y_ = tf.placeholder(tf.float32,[None,2])
        
        
        saver = tf.train.Saver()       
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
        
        image_data = vggMakeData.ImageData(vggMakeData.tfRecorf_test_path, BATCH_SIZE)
        image_data.read_tfRecord()
        img_batch,label_batch = image_data.get_data()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
#            print(ckpt)
#            print(vgg16Backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:                
                saver.restore(sess,ckpt.model_checkpoint_path)
                
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
               
                for index in range(STEP):
                    
                    Debug.print_current_value(index)
                    x_data, y_data = sess.run([img_batch,label_batch])
                    Debug.print_error()
                    Debug.print_current_value(y_data)
                    reshape = np.reshape(x_data,[BATCH_SIZE,forward.IMAGE_WIDTH,
                                               forward.IMAGE_HEIGTH,forward.NUM_CHANNELS])
                    
                    accuracy_score = sess.run(accuracy,feed_dict={x:reshape, y_:y_data})
    #                Debug.print_current_value(accuracy_score)           #打印调试
                    print("准确率 %f" % accuracy_score)
                
                coord.request_stop()
                coord.join(threads)
            
            else:
                print('No checkpoint file found')
                return   

def main():

    Test()
    
if __name__ == "__main__":
    main()
    