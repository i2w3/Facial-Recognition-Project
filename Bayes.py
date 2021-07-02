import numpy as np
import pandas as pd
from DataProcess import combineSeqData, combineLBPSeqData
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Data = pd.read_csv('pdData.csv')
X = combineSeqData(Data)
y = np.array(Data['Face'].values)

# X1 76个特征
pca = PCA(n_components=76, svd_solver='auto',
          whiten=True).fit(X)
X1 = pca.transform(X)

# X2 LBP
X2 = combineLBPSeqData(Data)

# X3 LBP + PCA 76个特征
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

print("Naive Bayes：")
for name, data in (["Raw", X], ["PCA", X1], ["LBP", X2], ["LBP + PCA", X3]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(GaussianNB(), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


print("Bernoulli Bayes：")
for name, data in (["Raw", X], ["PCA", X1], ["LBP", X2], ["LBP + PCA", X3]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(BernoulliNB(), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
