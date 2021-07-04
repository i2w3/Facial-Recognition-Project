import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData, combineLBPSeqData
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
Data = pd.read_csv('../pdData.csv')
X = combineLBPSeqData(Data)  # 使用LBP进行特征提取
y = np.array(Data['Face'].values)  # 读取分类类型的标签

clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_leaf=1)

# 搜索最佳深度并画图
depths = range(3, 20)
score = np.zeros(20)

f1_score_train=np.zeros(50)
f1_score_test=np.zeros(50)

num_folds = 10
scoring = 'accuracy'
for depth in depths:
    clf = RandomForestClassifier(max_depth=depth, min_samples_leaf=1)
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(clf, kafang_X, y, cv=kfold, scoring=scoring)
    score[depth] = cv_results.mean()
    f1_score_test[depth] = metrics.f1_score(clf.predict(X_test), y_test)
    f1_score_train[depth] = metrics.f1_score(clf.predict(X_train), y_train)

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
# data = pd.DataFrame({"score": score, "f1_score_train": f1_score_train, "f1_score_test": f1_score_test})
data = pd.DataFrame({"score": score})
sns.lineplot(data=data)
plt.xticks(np.linspace(0,20,21,endpoint=True))  # 设置x轴刻度
plt.xlabel("tree_depth")
plt.ylabel("score")
plt.title("scores varies with tree depths")
