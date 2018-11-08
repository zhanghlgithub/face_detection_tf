import cv2
import os
import shutil
import numpy as np
import debug
import tensorflow as tf

Debug = debug.DebugTool(True) #定义一个调试工具

PATH_TFR_TRAIN = "./data/train_data/mnist_train.tfrecords"
PATH_TFR_TEST = "./data/test_data/mnist_test.tfrecords"
PATH_TRAIN = "D:/ZhangHonglu/DeepLearningCode/VGG16_face_II/train_picture"
PATH_TEST = "D:/ZhangHonglu/DeepLearningCode/VGG16_face_II/test_picture"

PATH_IMG_1 = "D:/ZhangHonglu/DataSet/Face_Dataset/Positive_Sample/Sample_Set_1__face_128x96"    #train positive dataSet
PATH_IMG_2 = "D:/ZhangHonglu/DataSet/Face_Dataset/Negative_Sample/Sample_Set_1_128x96"          #train negative dataSet
PATH_IMG_3 = "D:/ZhangHonglu/DataSet/Face_Dataset/Test_Sample"                                #test dataSet
PATH_IMG_4 = "D:/ZhangHonglu/DataSet/Face_Dataset/App_Sample"

class MakeData():
    ''' function：制作训练和测试的数据集 '''
    def __init__(self,imgae_path,tfRecordName, batch_size, image_size,label_num):
        '''image_path: 图像的路径; batch_size: 每次喂入神经网络的图像数量;
         num: 总样本数; image_size: 图像的尺寸,label_vec:标签'''
        self.path = imgae_path
        self.tfRecordName = tfRecordName
        self.batch_size = batch_size
        self.image_size = image_size 
        self.label_vec = [0] * label_num
        self.count = 0
        
    def write_tfRecord(self):
        '''function: 把数据和对应标签制作成tfRecord文件'''
        writer = tf.python_io.TFRecordWriter(self.tfRecordName)
        
        dirImage = os.listdir(self.path)
        for index in dirImage:
            self.label_vec[self.count] = 1
            Debug.print_current_value(self.label_vec)
            Debug.print_current_value(index)
            for img in os.listdir(self.path + "/" + index):
#                Debug.print_current_value(img)
                src = cv2.imread(self.path + "/" + index + "/" + img)
                src = cv2.resize(src,self.image_size, interpolation=cv2.INTER_LINEAR)           #将读出的图片转换成指定大小
    #            src = cv2.GaussianBlur(src, (5,5), 0)           #通过高斯滤波来降噪
                src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
                src = cv2.equalizeHist(src)                     #图像的均值化操作2018.11.4号 
                img_raw = src.tobytes()
                labels = self.label_vec
                example = tf.train.Example(features=tf.train.Features(feature={
                            'img_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=labels))                    
                        }))
                writer.write(example.SerializeToString())
            self.label_vec[self.count] = 0
            self.count += 1
        writer.close()
        print("write tfrecord successful")
            
            
    def get_mnist(self):
        '''function: 获取数据集'''
        filename_queue = tf.train.string_input_producer([self.tfRecordName])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                                   'label' : tf.FixedLenFeature([len(self.label_vec)], tf.int64),
                                                   'img_raw' : tf.FixedLenFeature([], tf.string)
                                                   })
        
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img.set_shape([self.image_size[0] * self.image_size[1]])
        img = tf.cast(img, tf.float32) * (1. / 255)
        label = tf.cast(features['label'],tf.float32)
        
        return img, label
        
def make_data(imgae_path,tfRecordName, batch_size, image_size,label_vec):
    '''function: path:图片数据的路径; batch_size:一次喂给网络的图片个数;
        sample_num:样本总数; label_vec:标签向量'''
    Data = MakeData(imgae_path,tfRecordName, batch_size, image_size,label_vec)  #生成对象
        
    if os.path.exists(tfRecordName) is False:
        Debug.print_error()
        Data.write_tfRecord()
   
    train_image,label_image = Data.get_mnist() 
    
    image_batch,label_batch = tf.train.shuffle_batch([train_image,label_image],
                                                     batch_size=batch_size,
                                                     num_threads=2,
                                                     capacity = 10,
                                                     min_after_dequeue= 7 )
    return image_batch,label_batch


def resizeImage(path,resize,dst_path):
    '''Function:改变path目录下所有图片的大小规格 \
       path:图片的路径；resize：重置的大小;dst_path:重置文件要保存的路径 '''
    dir_image = os.listdir(path)
    Debug.print_current_value(len(dir_image))
    for img in dir_image:
        Debug.print_current_value(img)                      #调试打印
        
        src_path = path + "/" + img                         #需读取图片的绝对路径
        if os.path.getsize(src_path) == 0:                  #判断当前文件大小是否为0
            os.remove(src_path)                             #删除当前文件
            continue
        src = cv2.imread(src_path)   
        Debug.print_current_value(len(src))
        dst = cv2.resize(src, resize)
        if not os.path.exists(dst_path):                    #判断保存的【文件夹】是否存在,不存在则创建
            os.makedirs(dst_path)
            Debug.print_current_value(dst_path)             #调试打印
        dst_name = dst_path + "/" + img                     #保存到文件中的文件的名字
        cv2.imwrite(dst_name,dst)

def copyPicture(path,dst_path):    
    '''function:将path目录下的文件拷贝到dst_path目录下'''
    i = 0
    dirParent = os.listdir(path)
    for dirP in dirParent:
        
        PATN_SON = path + "/" + dirP                         #子目录
        dirSon = os.listdir(PATN_SON)
        for dirS in dirSon:
            file_name = str(i)
            oldname = PATN_SON + "/" + dirS 
            newname = dst_path + file_name + ".jpg"
            shutil.copyfile(oldname,newname)

def rename(path):
    '''function:对文件进行重新命名，path：当前的目录'''
    i = 0
    dir_file = os.listdir(path)
    for file_name in dir_file:
        old_name = path + "/" + file_name
        i += 1
        new_name = path + "/" + str(i) + ".jpg"      
        os.rename(old_name,new_name)

def face_detection(lib_path,image_path,dst_path):
    '''function:利用opencv库中的人脸识别工具截取图片中的人脸
        lib_path:通过级联分类器训练好的人脸分类器,
        image_path:待检测图像的目录,dst_path:检测到的人脸将要保存的路径'''
    face_classify = cv2.CascadeClassifier(lib_path)
    if face_classify.empty():
        Debug.print_error                                   #调试打印
    dir_image = os.listdir(image_path)
    for img in dir_image:
        src_path = image_path + "/" + img                   #图片的绝对路径
        src_image = cv2.imread(src_path)
        '''以下 2 行代码属于图像的形态学操作'''
        gray = cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY)   #将彩色图像转换成灰度图像
        gray = cv2.equalizeHist(gray)                       #将灰度图像进行均值化操作已增强特征点
        face = face_classify.detectMultiScale(gray, 1.3, 5) #人脸检测，将人脸区域的四个坐标点赋给face
        Debug.print_current_value(face)                     #调试打印
        for (x,y,w,h) in face:
            
            roi_image = src_image[y-8:y+h+16, x:x+w]
#            cv2.rectangle(src_image,(x, y),(x+w, y+h), (255,0,0), 2)
            if not os.path.exists(dst_path):                #判断保存的【文件夹】是否存在,不存在则创建
                os.makedirs(dst_path)    
                Debug.print_current_value(dst_path)         #调试打印
            cv2.imwrite(dst_path+"/"+ img,roi_image)        #将检测到的人脸保存到指定的文件目录下
#            cv2.imwrite(dst_path+"/"+ img,src_image)
           
#def main():
#    src_path = "D:/ZhangHonglu/DataSet/Face_Dataset/Positive_Sample/Sample_Set_3"
#    dst_path = "D:/ZhangHonglu/DataSet/Face_Dataset/Positive_Sample/Sample_Set_3__face_128x96"
#    lib_path = "D:/ZhangHonglu/DeepLearningCode/opencv_lib/haarcascades/haarcascade_frontalface_alt.xml"
#    rename(dst_path)
#    resizeImage(src_path,(128,96),dst_path)   
#    face_detection(lib_path,src_path,dst_path)
#    train_image,test_image = make_data(PATH_TRAIN,PATH_TFR_TRAIN, 10, (128,96), 2)
    
#if __name__ == "__main__":
#    main()
    
