
import tensorflow as tf
import vgg16Forward
import vgg16Backward
import vgg16MakeData
import debug
import numpy as np
DEBUG_PRINT = True
BATCH_SIZE = 100
STEP = 10
SAMPLE_NUM = 1000
#import cv2

Debug = debug.DebugTool(True)

def Test():
     
    with tf.Graph().as_default() as g:
        
        x = tf.placeholder(tf.float32,[BATCH_SIZE,vgg16Forward.IMAGE_WIDTH,
                                       vgg16Forward.IMAGE_HEIGTH,vgg16Forward.NUM_CHANNELS])
    
        y = vgg16Forward.forward(x,False,None)
        y_ = tf.placeholder(tf.float32,[None,2])
        
        ema = tf.train.ExponentialMovingAverage(vgg16Backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)       
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
        
        img_batch,label_batch = vgg16MakeData.make_data(vgg16MakeData.PATH_TEST,vgg16MakeData.PATH_TFR_TEST, 
                                                        BATCH_SIZE, 
                                                        (vgg16Forward.IMAGE_WIDTH,vgg16Forward.IMAGE_HEIGTH), 
                                                        vgg16Forward.FC_OUTPUT_NODE)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(vgg16Backward.MODEL_SAVE_PATH)
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
                    reshape = np.reshape(x_data,[BATCH_SIZE,vgg16Forward.IMAGE_WIDTH,
                                               vgg16Forward.IMAGE_HEIGTH,vgg16Forward.NUM_CHANNELS])
                    Debug.print_error()
                    
                    Debug.print_error()
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
    