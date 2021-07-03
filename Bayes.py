import numpy as np
import pandas as pd
from DataProcess import combineSeqData, combineLBPSeqData
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.decomposition import PCA

Data = pd.read_csv('pdData.csv')
X = combineSeqData(Data)
y = np.array(Data['Face'].values)

# X1 27个特征
pca = PCA(n_components=27, svd_solver='auto',
          whiten=True).fit(X)
X1 = pca.transform(X)

# X2 LBP
X2 = combineLBPSeqData(Data)

# X3 LBP + PCA 27个特征
pca = PCA(n_components=27, svd_solver='auto',
          whiten=True).fit(X2)
X3 = pca.transform(X2)

num_folds = 10
scoring = 'accuracy'

print("Naive Bayes：")
print("The face label of the dataset contains:", list(set(Data['Face'])))
for name, data in (["Raw", X], ["PCA", X1], ["LBP", X2], ["LBP + PCA", X3]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(GaussianNB(), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

DataNF = pd.read_csv('pdDataNF.csv')
X4 = combineSeqData(DataNF)
y4 = np.array(DataNF['Face'].values)

# X5 27个特征
pca = PCA(n_components=27, svd_solver='auto',
          whiten=True).fit(X4)
X5 = pca.transform(X4)

# X6 LBP
X6 = combineLBPSeqData(DataNF)

# X7 LBP + PCA 27个特征
pca = PCA(n_components=27, svd_solver='auto',
          whiten=True).fit(X6)
X7 = pca.transform(X6)

num_folds = 10
scoring = 'accuracy'

print("Combined funny&smiling Naive Bayes：")
print("The face label of the dataset contains:",list(set(DataNF['Face'])))
for name, data in (["Raw", X4], ["PCA", X5], ["LBP", X6], ["LBP + PCA", X7]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(GaussianNB(), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
