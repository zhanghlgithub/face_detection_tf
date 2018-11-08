
import sys
import os

class DebugTool():
    '''function：封装一些调试常用的函数'''
    
    def __init__(self,choise):
        '''function:初始化参数; choise：未T或F表示是否启用打印调试'''
        self.choise = choise
    
    def print_error(self):
        '''function:打印当前报错位于哪个函数的哪一行'''
        if self.choise:
#            file_name = os.path.basename(sys._getframe().f_code.co_filename) #获取当前运行文件的函数名
            func_name = sys._getframe().f_back.f_code.co_name   #获取调用函数名
            line_number = sys._getframe().f_back.f_lineno       #获取行号
            
            print("【%s】 Function Happen Erron At Line --%d--" %
                           (func_name, line_number))
        else:
            #do nothing
            pass
    
    def print_current_value(self,value):
        '''function:打印位于某个函数的某一行的输出值：'''
        if self.choise:
            
#            file_name = os.path.basename(sys._getframe().f_code.co_filename)  #获取当前运行文件的函数名
            func_name = sys._getframe().f_back.f_code.co_name   #获取调用函数名
            line_number = sys._getframe().f_back.f_lineno       #获取行号
            
            print(func_name + " Function " + " At Line --"+ str(line_number) +
                          "--"+ " outputResult is : " + str(value))
        else:
            #do nothing 
            pass
 
def get_file():
    
    return os.path.basename(sys._getframe().f_code.co_filename)
    
