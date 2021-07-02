import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Data = pd.read_csv('pdData.csv')
X = combineSeqData(Data)
y = np.array(Data['Face'].values)

pca = PCA(n_components=76, svd_solver='auto',
          whiten=True).fit(X)
X1 = pca.transform(X)

sc = StandardScaler()
sc.fit(X)
X2 = sc.transform(X)

pca = PCA(n_components=76, svd_solver='auto',
          whiten=True).fit(X2)
X3 = pca.transform(X2)

Kmeans = 1
# 74 0.601884
# 75 0.607680
# 76 0.608690
# 77 0.609195
# 78 0.607679

num_folds = 10
scoring = 'accuracy'

for name, data in (["Raw", X], ["PCA", X1], ["Std", X2], ["Std+PCA", X3]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(GaussianNB(), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

pca2 = PCA(n_components=114, svd_solver='auto',
          whiten=True).fit(X)
X4 = pca2.transform(X)

for name, data in (["Raw", X], ["PCA", X1], ["PCAN", X4]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(BernoulliNB(), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
