# How to download
下载并安装[git](https://git-scm.com/downloads)

再去命令行输入以下命令
```git
git clone git@github.com:i2w3/pypy.git
```
下载太慢？试试Github镜像站
```
git clone https://ghproxy.com/https://github.com/i2w3/pypy.git
```
# How to use
## 0.install requirements
```cmd
pip install pillow
pip install opencv-python
conda install numpy
conda install pandas
conda install seaborn
conda install scikit-learn
conda install scikit-image
```
## 1.Run build_pdData.py
生成pdData.csv(可以不生成，用自带的，都是一样的)，该文件类似这样：

|Seq|Sex|Age|Race|Face|Prop|
|----|----|----|----|----|----|
|1223|male|child|white|smiling|NoProp|
|1224|male|child|white|serious|NoProp|
|......|......|......|......|......|......|

主要用来存储预处理后的数据的序号(Seq)、性别(Sex)、Age(年龄)、肤色(Rase)、表情(Face)、是否佩戴道具(Prop)

除此之外，还会生成pdDataNF.csv，该文件是把Face标签中的"funny"改为"smiling"

对该文件的调用：
```python
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

# 需要比较合并funny和smiling后的分类效果用这个
DataNF = pd.read_csv('pdDataNF.csv')
y2 = np.array(Data['Face'].values)  # 读取分类类型的标签
```

## 2.Run Models
Bayes.py(贝叶斯)

DT&RT.py(决策树及随机森林)

KNN.py(K近邻)
