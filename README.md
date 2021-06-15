# How to use
## 1.Run build_pdData.py
生成pdData.csv，该文件类似这样：

|Seq|Sex|Age|Race|Face|Prop|
|----|----|----|----|----|----|
|1223|male|child|white|smiling|NoProp|
|1224|male|child|white|serious|NoProp|
|......|......|......|......|......|......|

主要用来存储预处理后的数据的序号(Seq)、性别(Sex)、Age(年龄)、肤色(Rase)、表情(Face)、是否佩戴道具(Prop)

对该文件的调用：
```python
import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData

Data = pd.read_csv('pdData.csv')

seq =np.array(Data['Seq'], dtype=str)  # 读取pdData.csv中的序号

# 主要用的是这两个
X = combineSeqData(Data)  # 根据序号读取图像数据（已扁平化）
y = np.array(Data['Face'].values)  # 读取分类类型的标签

images = combineSeqFid(Data)  # 根据序号读取图像（128 * 128）
target_names = list(set(Data['Face'])) # 显示标签的总类
```

## 2.模型展示