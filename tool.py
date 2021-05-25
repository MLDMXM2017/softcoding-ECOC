import numpy as np
import time
import os

def cache_matrix(matrix, file_name, mode = 'w'):
    out = open(file_name, mode)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])-1):
            out.writelines(str(matrix[i][j]) + ',')
        out.writelines(str(matrix[i][-1]))
        out.writelines('\n')

def cache_log(log, file_name, mode = 'a'):
    out = open(file_name, mode)
    out.writelines(log + '\n')

def get_time_str():
    return time.strftime('%m%d-%H%M%S',time.localtime(time.time()))

def mkdir(path):
    path=path.strip().rstrip("\\")
    os.makedirs(path)
    return path

