
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
 
from tensorflow.python.ops import control_flow_ops # 数据流控制库
# 从csv文件读取数据
df_TSMC = pd.read_csv('TSMC.csv')
df_acer = pd.read_csv('acer.csv')
df_AUO = pd.read_csv('AUO.csv')
df_SI2407 = pd.read_csv('SI2407.csv')

# 3个数据集的长度
# len(df_acer), len(df_TSMC), len(df_AUO)

# *获取关闭的数据列*
# 收盘价：股票收盘价
# -用于预测股票收盘价
# p = df_acer['Close'].values.astype('float32')
p = df_SI2407['Close'].values.astype('float32')
print(p)
