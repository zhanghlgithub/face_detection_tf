# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:59:27 2018

@author: zhl
"""
import cv2
import os
import shutil
import numpy as np
import debug
import tensorflow as tf
from random import shuffle

Debug = debug.DebugTool(True) #定义一个调试工具

pre_path = "D:/ZhangHonglu/DeepLearningCode/vgg-face/vgg-face/pre_pricture"
tfRecorf_path = "./tfRecord_data/train_data/train.tfrecords"
train_image = "./train_image"

pre_test_path = "D:/ZhangHonglu/DeepLearningCode/vgg-face/vgg-face/pre_test_picture"
tfRecorf_test_path = "./tfRecord_data/test_data/test.tfrecords"
test_image = "./test_image"

class MakeData():
    ''' function：制作训练和测试的数据集 '''
    def __init__(self, pre_path, image_path, tfRecordName, image_size):
        '''
        Arg:
            - pre_path: 预处理图像路径
            - image_path: 源图像路径
            - tfRecordName:tfRecord文件保存路径
            - image_size: 图像的尺寸,
        '''
        self.path = pre_path
        self.image_path = image_path
        self.tfRecordName = tfRecordName
        self.image_size = image_size 
        self.count = 0
        self.labels = 0
        self.data_txt = "./data.txt"
        
    def makeTxt(self):
        '''function:把图像名和对应标签对应写入txt文件'''        
        txt_list = []
        for dirname in os.listdir(self.path):
            self.labels += 1
            Debug.print_current_value(self.labels)
            Debug.print_current_value(dirname)
            for filename in os.listdir(self.path + "/" + dirname):
                merge_data = filename + " " + str(self.labels) + "\n"
#                Debug.print_current_value(merge_data)
                txt_list.append(merge_data)
        
        #随机打乱txt_list的值
        shuffle(txt_list)
        fp = open(self.data_txt,'w')
        fp.truncate()   #清空文件内容
        fp.close()
        fp = open(self.data_txt,'a')
        for i in range(len(txt_list)):
            fp.write(txt_list[i])
        fp.close()
    
    def write_tfRecord(self):
        '''function: 把数据和对应标签制作成tfRecord文件'''
        writer = tf.python_io.TFRecordWriter(self.tfRecordName)
        
        fp = open(self.data_txt, "r")
        contents = fp.readlines()
        fp.close()
        for content in contents:
#            Debug.print_current_value(content)
            value = content.split()
            image_path = os.path.join(self.image_path, value[0])
#            Debug.print_current_value(image_path)
            src = cv2.imread(image_path)
            src = cv2.resize(src,self.image_size, interpolation=cv2.INTER_LINEAR)           #将读出的图片转换成指定大小
#            src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)      #图像转换成灰度图像
#            src = cv2.equalizeHist(src)                     #图像的均值化操作2018.11.4号 
#            Debug.print_current_value(src)
            img_raw = src.tobytes()
            Debug.print_current_value(value[1])
            labels = [0] * 2
            labels[int(value[1]) - 1] = 1
            Debug.print_current_value(labels)
            example = tf.train.Example(features=tf.train.Features(feature={
                            'img_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=labels))                    
                        }))
            writer.write(example.SerializeToString())
                
        writer.close()
        print("write tfrecord successful")
    
#make_data = MakeData(pre_test_path, test_image, tfRecorf_test_path, (250, 250))
#make_data.makeTxt()  
#make_data.write_tfRecord()   
            
class ImageData():
            
    def __init__(self, tfRecordName,batch_szie):
        self.tfRecordName = tfRecordName
        self.batch_size = batch_szie
        
    def read_tfRecord(self):
        '''function: 读取tfRecord文件'''
        filename_queue = tf.train.string_input_producer([self.tfRecordName])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                                   'label' : tf.FixedLenFeature([2], tf.int64),
                                                   'img_raw' : tf.FixedLenFeature([], tf.string)
                                                   })
        
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img.set_shape([250 * 250 * 3])
        img = tf.cast(img, tf.float32) 
        self.image = img / 127.5 - 1           #图像的数值位于[-1,1]
        self.labels = tf.cast(features['label'],tf.float32)
    
    def get_data(self):
        image_batch,label_batch = tf.train.shuffle_batch([self.image,self.labels],
                                             batch_size=self.batch_size,
                                             num_threads=2,
                                             capacity = 100,
                                             min_after_dequeue= 70 )
        return image_batch, label_batch

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
        new_name = path + "/nor_face_" + str(i) + ".jpg" 
        
        os.rename(old_name,new_name)
        
#rename(os.path.join(pre_test_path,"2_nor_face"))

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
     
    

