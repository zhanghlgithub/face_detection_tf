
import tensorflow as tf
import vgg16Forward
import vgg16Backward
#import vgg16MakeData
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
        
        x = tf.placeholder(tf.float32,[BATCH_SIZE,vgg16Forward.IMAGE_WIDTH,
                                       vgg16Forward.IMAGE_HEIGTH,vgg16Forward.NUM_CHANNELS])
    
        y = vgg16Forward.forward(x,False,None)
#        predict = tf.argmax(y, 1)
        
        ema = tf.train.ExponentialMovingAverage(vgg16Backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)       
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(vgg16Backward.MODEL_SAVE_PATH)
            print(ckpt)
            print(vgg16Backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
#                print(ckpt.model_checkpoint_path)                
                    reshape = np.reshape(mnist,[BATCH_SIZE,vgg16Forward.IMAGE_WIDTH,
                                           vgg16Forward.IMAGE_HEIGTH,vgg16Forward.NUM_CHANNELS])

                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score = sess.run(y,feed_dict={x:reshape})
#                    print("准确率 %g" % accuracy_score)
                    print(accuracy_score)
            else:
                print('No checkpoint file found')
                return   

def main():
    
    src = cv2.imread("D:/ZhangHonglu/DeepLearningCode/VGG16_face_II/2.jpg")
    src = cv2.resize(src,(128,96), interpolation=cv2.INTER_LINEAR)           #将读出的图片转换成指定大小
    src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    src = cv2.equalizeHist(src) 
    Test(src)
    
if __name__ == "__main__":
    main()
    