import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

Data = pd.read_csv('../pdData.csv')
y = np.array(Data['Face'].values)  # 读取分类类型的标签

dist = Counter(y)  # 记录各标签的数目
x1 = list(dist.keys())
y1 = list(dist.values())

explode = (0.05, 0, 0)
colors = ['steelblue', 'gold', 'mediumturquoise']

plt.pie(y1, labels=x1, explode=explode, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=100)

plt.axis('equal')
plt.savefig('./plotDataSet.png', bbox_inches='tight')
plt.show()
