
import tensorflow as tf 
import vgg16Forward
import vgg16MakeData
import numpy as np
import os
import debug

Debug = debug.DebugTool(False) #定义一个调试工具

BATCH_SIZE = 10
LEARNING_RATE_BASE = 0.001 #学习率
LEARNING_RATE_DECAY = 0.99 #学习衰减率
REGULARIZER = 0.0001
STEPS = 2000 
MOVING_AVERAGE_DECAY = 0.99 #滑动平均值的衰减率
SAMPLE_NUM = 10000

MODEL_SAVE_PATH = "./model"
MODEL_NAME = "smoke_model"

def backward():
    
    X = tf.placeholder(tf.float32,shape=[BATCH_SIZE,vgg16Forward.IMAGE_WIDTH,
                                         vgg16Forward.IMAGE_HEIGTH,vgg16Forward.NUM_CHANNELS])
    y_real = tf.placeholder(tf.float32,shape=[None,vgg16Forward.FC_OUTPUT_NODE]) #X对应的真实标签
    y_predict = vgg16Forward.forward(X, True, REGULARIZER) #X对应的预测值
    
    print("前向传播网络搭建完毕")
    
    global_step = tf.Variable(0,trainable=False)
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict,labels=tf.argmax(y_real,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    #使用梯度下降法优化损失函数loss，用指数衰减的方式更改学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            (SAMPLE_NUM * 2) / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True
            )
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    #给权重w设定影子值
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name="train")
        
    saver = tf.train.Saver()
    Debug.print_error()
    img_batch,label_batch = vgg16MakeData.make_data(vgg16MakeData.PATH_TRAIN,vgg16MakeData.PATH_TFR_TRAIN, BATCH_SIZE, 
                                                    (vgg16Forward.IMAGE_WIDTH,vgg16Forward.IMAGE_HEIGTH), 
                                                    vgg16Forward.FC_OUTPUT_NODE)
    
    #创建会话sess
    Debug.print_error()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
#        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
#        if ckpt and ckpt.model_checkpoint_path:
#            saver.restore(sess, ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for index in range(STEPS):
            x_data, y_data = sess.run([img_batch,label_batch])
            
            Debug.print_current_value(y_data.shape)
            reshape_xs = np.reshape(x_data,[BATCH_SIZE,vgg16Forward.IMAGE_WIDTH,
                                              vgg16Forward.IMAGE_HEIGTH,vgg16Forward.NUM_CHANNELS])
            Debug.print_error()
            _, loss_value, step = sess.run([train_op, loss, global_step], 
                                           feed_dict={X:reshape_xs, y_real:y_data})
            
            #每训练100轮保存当前的模型
            if (index) % 20 == 0:
                print("After %d training steps,loss on training batch is %g." % (step, loss_value))
                #把当前会话保存到指定的路径下面
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                           global_step=global_step)
        
        coord.request_stop()
        coord.join(threads)
        
def main():
    backward()

if __name__ == "__main__":
    main()    
    
    