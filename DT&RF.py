import numpy as np
import pandas as pd
from DataProcess import combineLBPSeqData
from sklearn import model_selection
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

Data = pd.read_csv('pdData.csv')
X = combineLBPSeqData(Data)  # 使用LBP进行特征提取
y = np.array(Data['Face'].values)

# pca
pca = PCA(n_components=76, svd_solver='auto',
          whiten=True).fit(X)
X1 = pca.transform(X)

# 建立模型（决策树/随机森林）
clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_leaf=1)
clf1 = RandomForestClassifier(max_depth=6)

print("Decision Tree:")
num_folds = 10
scoring = 'accuracy'
for name, data in (["RAW", X], ["PCA", X1]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(clf, data, y, cv=kfold, scoring=scoring)
    msg = "%s DT: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print("Random Forest:")
num_folds = 10
scoring = 'accuracy'
for name, data in (["RAW", X], ["PCA", X1]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(clf1, data, y, cv=kfold, scoring=scoring)
    msg = "%s DT: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
