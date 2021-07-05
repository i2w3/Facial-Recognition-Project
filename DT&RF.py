import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData,combineLBPSeqData
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


Data = pd.read_csv('pdData.csv')
raw_X = combineSeqData(Data)#原始数据
lbp_X = combineLBPSeqData(Data)#使用LBP进行特征提取后的数据
y = np.array(Data['Face'].values)



#pca
pca = PCA(n_components=76, svd_solver='auto',
          whiten=True).fit(raw_X)
pca_X = pca.transform(raw_X)



#建立模型（决策树/随机森林）
clf0 = DecisionTreeClassifier(max_depth=5,min_samples_leaf=1)
clf1 = RandomForestClassifier(max_depth=6)

num_folds = 10
scoring = 'accuracy'
for name, data in (["RAW", raw_X], ["PCA", pca_X], ["LBP", lbp_X]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results0 = model_selection.cross_val_score(clf0, data, y, cv=kfold, scoring=scoring)#决策树
    cv_results1 = model_selection.cross_val_score(clf1, data, y, cv=kfold, scoring=scoring)#随机森林
    msg0 = "%s DT: %f (%f)" % (name, cv_results0.mean(), cv_results0.std())
    print(msg0)
    msg1 = "%s RF: %f (%f)" % (name, cv_results1.mean(), cv_results1.std())
    print(msg1)






