import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData

Data = pd.read_csv('pdData.csv')

seq = np.array(Data['Seq'], dtype=str)  # 读取pdData.csv中的序号

# 主要用的是这两个
X = combineSeqData(Data)  # 根据序号读取图像数据（已扁平化）
y = np.array(Data['Face'].values)  # 读取分类类型的标签

images = combineSeqFid(Data)  # 根据序号读取图像（128 * 128）
target_names = list(set(Data['Face']))  # 显示标签的总类

print("第一个数据的序号为", seq[0], "\n图像数据是", X[0], "\n标签是", y[0])
